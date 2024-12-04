from abc import abstractmethod

import numpy as np

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene

import logging
from abc import ABC


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


class GripperActionMode(ABC):

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene)
        self.action_post_step(scene, action)

    def action_step(self, scene: Scene):
        scene.step()

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    @abstractmethod
    def action_bounds(self):
        pass


class Discrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

    def _actuate(self, scene, action):
        done = False
        while not done:
            done = scene.robot.gripper.actuate(action, velocity=0.2)
            scene.pyrep.step()
            scene.task.step()

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        open_condition = all(
            x > 0.9 for x in scene.robot.gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            done = False
            if not self._detach_before_open:
                self._actuate(scene, action)
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    scene.robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                scene.robot.gripper.release()
            if self._detach_before_open:
                self._actuate(scene, action)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([1])



class GripperJointPosition(GripperActionMode):
    """Control the target joint positions absolute or delta) of the gripper.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.

    Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(scene.robot))
        action = action.repeat(2)  # use same action for both joints
        a = action if self._absolute_mode else np.array(
            scene.robot.gripper.get_joint_positions())
        scene.robot.gripper.set_joint_target_positions(a)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.gripper.set_joint_target_positions(
            scene.robot.gripper.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])
    

class BimanualGripperJointPosition(GripperJointPosition):
    
    def action_pre_step(self, scene: Scene, action: np.ndarray):

        if not self._control_mode_set:
            scene.robot.right_gripper.set_control_loop_enabled(True)
            scene.robot.left_gripper.set_control_loop_enabled(True)
            self._control_mode_set = True

        assert_action_shape(action, self.action_shape(scene.robot))

        right_action = action[:1].repeat(2) 
        left_action = action[1:].repeat(2) 

        if not self._absolute_mode:
            right_action = right_action + np.array(scene.robot.gripper.get_joint_positions())
            left_action = left_action + np.array(scene.robot.gripper.get_joint_positions())
            
        scene.robot.right_gripper.set_joint_target_positions(right_action)
        scene.robot.left_gripper.set_joint_target_positions(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.right_gripper.set_joint_target_positions(
            scene.robot.right_gripper.get_joint_positions())
        
        scene.robot.left_gripper.set_joint_target_positions(
            scene.robot.left_gripper.get_joint_positions())        
    
    def action_shape(self, scene: Scene) -> tuple:
        return 2,



class UnimanualDiscrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 robot_name: str = 'left'):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self.robot_name = robot_name


    def _actuate(self, scene, action):
        done = False
        while not done:
            if self.robot_name == 'right':
                done = scene.robot.right_gripper.actuate(action, velocity=0.2)
            elif self.robot_name == 'left':
                done = scene.robot.left_gripper.actuate(action, velocity=0.2)
            else:
                done = scene.robot.gripper.actuate(action, velocity=0.2)

            scene.pyrep.step()
            scene.task.step()

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        if self.robot_name == 'right':
            open_condition = all(x > 0.9 for x in scene.robot.right_gripper.get_open_amount())
        else:
            open_condition = all(x > 0.9 for x in scene.robot.left_gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            done = False
            if not self._detach_before_open:
                self._actuate(scene, action)
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    if self.robot_name == 'right':
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                if self.robot_name == 'right':
                    scene.robot.right_gripper.release()
                else:
                    scene.robot.left_gripper.release()
            if self._detach_before_open:
                self._actuate(scene, action)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()


    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])



class BimanualDiscrete(Discrete):
    
    def _actuate(self, scene, action):

        right_action = action[0]
        left_action = action[1]
        done = False
        right_done = False
        left_done = False

        while not done:
            if not right_done:
                right_done = scene.robot.right_gripper.actuate(right_action, velocity=0.2)
            if not left_done:
                left_done = scene.robot.left_gripper.actuate(left_action, velocity=0.2)
            done = right_done and left_done
            scene.pyrep.step()
            scene.task.step()

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)


        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        logging.warning("Object with name %s is already grasped by left robot", g_obj.get_name())
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                # If gripper close action, the check for grasp.                
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        logging.warning("Object with name %s is already grasped by right robot", g_obj.get_name())
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

    def action_shape(self, scene: Scene) -> tuple:
        return 2,

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])
