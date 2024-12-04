from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.arm import Arm
from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight
from pyrep.robots.end_effectors.gripper import Gripper

from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation import Observation
from rlbench.backend.observation import UnimanualObservationData
from rlbench.backend.observation import UnimanualObservation
from rlbench.backend.observation import BimanualObservation

from rlbench.backend.robot import Robot
from rlbench.backend.robot import UnimanualRobot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig

STEPS_BEFORE_EPISODE_START = 10

import logging


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None

        if self.robot.is_bimanual:
            self._start_arm_joint_pos = [robot.right_arm.get_joint_positions(), robot.left_arm.get_joint_positions()]
            self._starting_gripper_joint_pos = [robot.right_gripper.get_joint_positions(), robot.left_gripper.get_joint_positions()]
        else:
            self._start_arm_joint_pos = robot.arm.get_joint_positions()
            self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
    
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])

        self.camera_sensors = {camera_name: VisionSensor(f"cam_{camera_name}") for camera_name, _ in self._obs_config.camera_configs.items()}
        self.camera_sensors_mask = {camera_name: VisionSensor(f'cam_{camera_name}_mask') for camera_name, _ in self._obs_config.camera_configs.items()}


        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        # ..todo:: fixme convert to a list
        if self.robot.is_bimanual:
            self._initial_robot_state = [(robot.right_arm.get_configuration_tree(),
                                     robot.right_gripper.get_configuration_tree()),
                                     (robot.left_arm.get_configuration_tree(),
                                     robot.left_gripper.get_configuration_tree())]
        else:
            self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())

        self._ignore_collisions_for_current_waypoint = False

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        if self.robot.is_bimanual:
               self._robot_shapes = [self.robot.right_arm.get_objects_in_tree(object_type=ObjectType.SHAPE), 
               self.robot.left_arm.get_objects_in_tree(object_type=ObjectType.SHAPE)]
               self._right_execute_demo_joint_position_action = None
               self._left_execute_demo_joint_position_action = None
        else:
            self._robot_shapes = self.robot.arm.get_objects_in_tree(
                object_type=ObjectType.SHAPE)           
            self._execute_demo_joint_position_action = None

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot.release_gripper()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index
        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot.is_in_collision():
                        logging.error("robot is in collision")
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                logging.error('Error when checking waypoints. Exception is: %s', e)
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """

        self.robot.release_gripper()

        if self.robot.is_bimanual:
            self.reset_bimanual()
        else:
            self.reset_unimanual()

        self.robot.zero_velocity()
        
        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def reset_unimanual(self) -> None:
        arm, gripper = self._initial_robot_state   
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        
        self.robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)


    def reset_bimanual(self) -> None:

        for arm, gripper in self._initial_robot_state:        
            self.pyrep.set_configuration_tree(arm)
            self.pyrep.set_configuration_tree(gripper)
        
        self.robot.right_arm.set_joint_positions(self._start_arm_joint_pos[0], disable_dynamics=True)
        self.robot.right_gripper.set_joint_positions(self._starting_gripper_joint_pos[0], disable_dynamics=True)

        self.robot.left_arm.set_joint_positions(self._start_arm_joint_pos[1], disable_dynamics=True)
        self.robot.left_gripper.set_joint_positions(self._starting_gripper_joint_pos[1], disable_dynamics=True)


    def get_observation(self) -> Observation:

        observation_data = {}
        perception_data = {}

        # ..todo:: extract methods
        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        for camera_name, camera_config in self._obs_config.camera_configs.items():            

            rgb_data, depth_data, pcd_data = get_rgb_depth(self.camera_sensors[camera_name], camera_config.rgb, camera_config.depth, camera_config.point_cloud,
            camera_config.rgb_noise, camera_config.depth_noise, camera_config.depth_in_meters)

            if camera_config.mask and camera_config.masks_as_one_channel:
                mask_data = get_mask(self.camera_sensors_mask[camera_name], rgb_handles_to_mask)
            elif camera_config.mask:
                mask_data = get_mask(self.camera_sensors_mask[camera_name], lambda x: x)
            else:
                mask_data = None
                
            perception_data.update({f'{camera_name}_rgb': rgb_data, f'{camera_name}_depth': depth_data, f'{camera_name}_point_cloud': pcd_data,
                                     f'{camera_name}_mask': mask_data})
    



        def get_proprioception(arm: Arm, gripper: Gripper):
            tip = arm.get_tip()

            if self._obs_config.joint_velocities:
                joint_velocities=np.array(arm.get_joint_velocities())
                joint_velocities=self._obs_config.joint_velocities_noise.apply(joint_velocities)
            else:
                joint_velocities=None

            if self._obs_config.joint_positions:
                joint_positions = np.array(arm.get_joint_positions())
                joint_positions = self._obs_config.joint_positions_noise.apply(joint_positions)
            else:
                joint_positions = None
            
            if self._obs_config.joint_forces:
                fs = arm.get_joint_forces()
                vels = arm.get_joint_target_velocities()
                joint_forces = np.array([-f if v < 0 else f for f, v in zip(fs, vels)])
                joint_forces = self._obs_config.joint_forces_noise.apply(joint_forces)
            else:
                joint_forces=None

            if self._obs_config.gripper_open:
                if gripper.get_open_amount()[0] > 0.95:
                    gripper_open = 1.0
                else:
                    gripper_open = 0.0
            else:
                gripper_open = None

            if self._obs_config.gripper_pose:
                gripper_pose = tip.get_pose()
            else:
                gripper_pose = None


            if self._obs_config.gripper_matrix:
                gripper_matrix = tip.get_matrix()
            else:
                gripper_matrix = None

            if self._obs_config.gripper_touch_forces:
                ee_forces = gripper.get_touch_sensor_forces()
                ee_forces_flat = []
                for eef in ee_forces:
                    ee_forces_flat.extend(eef)
                gripper_touch_forces = np.array(ee_forces_flat)
            else:
                gripper_touch_forces =  None


            if self._obs_config.gripper_joint_positions:
                gripper_joint_positions= np.array(gripper.get_joint_positions())
            else:
                gripper_joint_positions = None


            if self._obs_config.record_ignore_collisions:
                if self._ignore_collisions_for_current_waypoint:
                    ignore_collisions = np.array(1.0)
                else:
                    ignore_collisions = np.array(0.0)
            else:
                ignore_collisions = None

            return {"joint_velocities": joint_velocities, 
            "joint_positions": joint_positions,
            "joint_forces": joint_forces, 
            "gripper_open": gripper_open,
            "gripper_pose": gripper_pose,
            "gripper_matrix": gripper_matrix,
            "gripper_touch_forces": gripper_touch_forces,
            "gripper_joint_positions": gripper_joint_positions, 
            "ignore_collisions": ignore_collisions}


        if self.robot.is_bimanual:
            observation_data["right"] = UnimanualObservationData(**get_proprioception(self.robot.right_arm, self.robot.right_gripper))
            observation_data["left"] = UnimanualObservationData(**get_proprioception(self.robot.left_arm, self.robot.left_gripper))
        else:
            observation_data.update(get_proprioception(self.robot.arm, self.robot.gripper))

        task_low_dim_state=(
            self.task.get_low_dim_state() if
            self._obs_config.task_low_dim_state else None),

        observation_data.update({
            "task_low_dim_state": task_low_dim_state,
            "perception_data": perception_data,
            "misc": self._get_misc()
        })

        if self.robot.is_bimanual:
            obs = BimanualObservation(**observation_data)
        else:
            obs = UnimanualObservation(**observation_data)

        obs = self.task.decorate_observation(obs)

        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def execute_waypoints_unimanual(self, do_record) -> bool:
        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue

                colliding_shapes = []                

                grasped_objects = self.robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                object_type=ObjectType.SHAPE) if s not in grasped_objects
                                and s not in self._robot_shapes and s.is_collidable()
                                and self.robot.arm.check_arm_collision(s)]
            

                logging.info("got list of colliding objects: %s", colliding_shapes)
                
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    logging.error("unable to get path %s", e)
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()

                logging.info("point.get_ext() %s", str(ext))

                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._execute_demo_joint_position_action = path.get_executed_joint_position_action()
                    do_record()
                    success, term = self.task.success()

                point.end_of_path()
                path.clear_visualization()
                logging.info("done executing path")

                if len(ext) > 0:
                    self._handle_extensions_strings(ext, do_record)
      

            if not self.task.should_repeat_waypoints() or success:
                return success


    def execute_waypoints_bimanual(self, do_record) -> bool:
        right_waypoints = self.task.right_waypoints
        left_waypoints = self.task.left_waypoints

        for i, right_point in enumerate(right_waypoints.copy()):
            ext = right_point.get_ext()
            if 'repeat' in ext:
                j = ext.rsplit('_', maxsplit=1)
                j = int(j[-1])
                for _ in range(j):
                    right_waypoints.insert(i, right_point)


        for i, left_point in enumerate(left_waypoints.copy()):
            ext = left_point.get_ext()
            if 'repeat' in ext:
                j = ext.rsplit('_', maxsplit=1)
                j = int(j[-1])
                for _ in range(j):
                    left_waypoints.insert(i, left_point)

        while len(left_waypoints) > len(right_waypoints):
            right_waypoints.append(right_waypoints[-1])

        while len(right_waypoints) > len(left_waypoints):
            left_waypoints.append(left_waypoints[-1])

        
        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            # ..fixme:: some waypoints might be skipped due to zip -> add dummy waypoints
            for i, (right_point, left_point) in enumerate(zip(right_waypoints, left_waypoints)):
                self._ignore_collisions_for_current_waypoint = right_point._ignore_collisions or left_point._ignore_collisions
                right_point.start_of_path()
                left_point.start_of_path()
                if right_point.skip or left_point.skip:
                    print("skipping waypoints!")
                    logging.error("skipping waypoints!")
                    continue
        
                grasped_objects = self.robot.right_gripper.get_grasped_objects() + self.robot.left_gripper.get_grasped_objects()
                colliding_shapes = []
                for s in self.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
                    if s in grasped_objects:
                        continue
                    #if s in self._robot_shapes:
                    #    continue
                    if not s.is_collidable():
                        continue
                    if self.robot.right_arm.check_arm_collision(s):
                        colliding_shapes.append(s)
                    elif self.robot.left_arm.check_arm_collision(s):
                        colliding_shapes.append(s)
                
                logging.debug("got list of colliding objects: %s", ", ".join([s.get_name()  for s in colliding_shapes]))
                
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    right_path = right_point.get_path()
                    left_path = left_point.get_path()
                except ConfigurationPathError as e:
                    logging.error("Unable to get path %s", e)
                    raise DemoError(f'Could not get a path for waypoint {right_point.name} or {left_point.name}.', task=self.task) from e
                finally:
                    [s.set_collidable(True) for s in colliding_shapes]

                right_ext = right_point.get_ext()
                left_ext = left_point.get_ext()

                right_path.visualize()
                left_path.visualize()

                right_done = False
                left_done = False
                success = False
                while not (right_done and left_done):
                    if not right_done and right_path.step():                
                        right_point.end_of_path()
                        right_path.clear_visualization()
                        for ext in right_ext.split(";"):
                            self._handle_extensions_strings(ext.strip(), do_record)
                        right_done = True

                    if not left_done and left_path.step():
                        left_point.end_of_path()
                        left_path.clear_visualization()
                        for ext in left_ext.split(";"):
                            self._handle_extensions_strings(ext.strip(), do_record)
                        left_done = True

                    self.step()
                    self._right_execute_demo_joint_position_action = right_path.get_executed_joint_position_action()
                    self._left_execute_demo_joint_position_action = left_path.get_executed_joint_position_action()
                    do_record()
                    success, term = self.task.success()

            if not self.task.should_repeat_waypoints() or success:
                return success


    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        demo = []

        def do_record():
            self._demo_record_step(demo, record, callable_each_step)

        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())

        success = False
        if self.robot.is_bimanual:
            success = self.execute_waypoints_bimanual(do_record)
        else:
            success = self.execute_waypoints_unimanual(do_record)
            

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                do_record()
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)
    
    def _handle_extensions_strings(self, ext, do_record):
        """
        Extensions strings are defined in the field under the 'Common Tab' when editing a waypoint
        """
        if len(ext) == 0:
            return

        contains_param = False
        start_of_bracket = -1
        name = ext.split('_', maxsplit=1)[0]
        if 'open_gripper(' in ext:
            self.robot.release_gripper(name)
            start_of_bracket = ext.index('open_gripper(') + 13
            contains_param = ext[start_of_bracket] != ')'
            if not contains_param:
                done = False
                while not done:
                    done = self.robot.actutate_gripper(1.0, 0.04, name)
                    self.pyrep.step()
                    self.task.step()
                    if self._obs_config.record_gripper_closing:
                        do_record()
        elif 'close_gripper(' in ext:
            start_of_bracket = ext.index('close_gripper(') + 14
            contains_param = ext[start_of_bracket] != ')'
            if not contains_param:
                done = False
                while not done:
                    done = self.robot.actutate_gripper(0.0, 0.04, name)
                    self.pyrep.step()
                    self.task.step()
                    if self._obs_config.record_gripper_closing:
                        do_record()

        if contains_param:
            rest = ext[start_of_bracket:]
            num = float(rest[:rest.index(')')])
            done = False
            logging.warning("not tested yet")
            while not done:
                done = self.robot.actutate_gripper(num, 0.04, name)
                self.pyrep.step()
                self.task.step()
                if self._obs_config.record_gripper_closing:
                    do_record()

        if 'close_gripper(' in ext:
            for g_obj in self.task.get_graspable_objects():
                self.robot.grasp(g_obj, name)
        do_record()

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            demo_list.append(self.get_observation())
        if func is not None:
            func(self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)


        for camera_name, camera_config in self._obs_config.camera_configs.items():
            _set_rgb_props(self.camera_sensors[camera_name], camera_config.rgb, camera_config.depth, camera_config)
   
            if camera_config.mask:
                _set_mask_props(
                self.camera_sensors_mask[camera_name],
                camera_config.mask,
                camera_config)
       

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)

    def _get_misc(self):
        misc = {}
        for camera_name, camera in self.camera_sensors.items():
            if camera.still_exists():
                misc.update({
                    f'{camera_name}_camera_extrinsics': camera.get_matrix(),
                    f'{camera_name}_camera_intrinsics': camera.get_intrinsic_matrix(),
                    f'{camera_name}_camera_near': camera.get_near_clipping_plane(),
                    f'{camera_name}_camera_far': camera.get_far_clipping_plane(),
                })
        misc.update({"variation_index": self._variation_index})
        if self.robot.is_bimanual and self._right_execute_demo_joint_position_action is not None:
            
            misc.update({"right_executed_demo_joint_position_action": self._right_execute_demo_joint_position_action,
                         "left_executed_demo_joint_position_action": self._left_execute_demo_joint_position_action})
            self._right_execute_demo_joint_position_action = None
            self._left_execute_demo_joint_position_action = None
        
        elif not self.robot.is_bimanual and self._execute_demo_joint_position_action is not None:
            misc.update({"executed_demo_joint_position_action": self._execute_demo_joint_position_action})
            self._execute_demo_joint_position_action = None
        return misc
