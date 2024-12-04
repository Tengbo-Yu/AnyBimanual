#!/usr/bin/env python3


import os
import traceback

from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperLeft, PandaGripperRight
from pyrep.objects.shape import Shape

from yarr.utils.video_utils import CircleCameraMotion
from yarr.utils.video_utils import TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.exceptions import *
from rlbench.observation_config import ObservationConfig
from rlbench.observation_config import CameraConfig

from rlbench.backend.robot import Robot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.scene import Scene

from rlbench.backend.task import BIMANUAL_TASKS_PATH
from rlbench.backend.const import BIMANUAL_TTT_FILE

import rich_click as click
from click_prompt import choice_option


from rlbench.utils import name_to_task_class

def main():
    render_videos_for_task()


def get_bimanual_tasks():
    return [t.replace('.py', '') for t in
    os.listdir(BIMANUAL_TASKS_PATH) if t != '__init__.py' and t.endswith('.py')]


@click.command()
@choice_option('--bimanual-task-files', type=click.Choice(sorted(get_bimanual_tasks())), multiple=True)
@click.option('--add-task-description/--no-task-description', default=False)
@click.option('--add-gripper-color/--no-gripper-color', default=False)
def render_videos_for_task(bimanual_task_files, add_task_description, add_gripper_color):

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    pr = PyRep()
    ttt_file = os.path.join(CURRENT_DIR, '..', 'rlbench', BIMANUAL_TTT_FILE)
    pr.launch(ttt_file, responsive_ui=True)
    pr.step_ui()

    robot = BimanualRobot(PandaRight(), PandaGripperRight(), PandaLeft(), PandaGripperLeft())

    if add_gripper_color:
        robot.colorize_gripper()

    cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                              render_mode=RenderMode.OPENGL3)
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.right_shoulder_camera = cam_config
    obs_config.left_shoulder_camera = cam_config
    obs_config.overhead_camera = cam_config
    obs_config.wrist_camera = cam_config
    obs_config.front_camera = cam_config

    scene = Scene(pr, robot, obs_config)

    for task_file in bimanual_task_files:


        if add_task_description:
            task_name_mapping = {}
            task_description = task_name_mapping.get(task_file, f"Demo for task {task_file}")
        else:
            task_description = ""

        task_class = name_to_task_class(task_file, True)
        task = task_class(pr, robot,  task_file.replace('.py', ''))
        try:
            print(f"Loading {task_file}")
            scene.load(task)
            pr.start()
            pr.step_ui()
            
            record_demo(scene, task, task_file, task_description)
            scene.reset()
            scene.unload()

            pr.stop()
            pr.step_ui()


        except Exception as e:
            print("error while loading task")
            print(e)


    pr.step_ui()

    pr.stop()
    pr.shutdown()


def record_demo(scene, task, task_file, task_description):
        
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam = VisionSensor.create([1920, 1200], background_color=[1.0, 1.0, 1.0])
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)

        scene_background = ["Wall1", "Wall2", "Wall3", "Wall4", "Floor", "Roof", "ResizableFloor_5_25_visibleElement"]

        for object_name in scene_background: 
            Shape(object_name).set_renderable(False)

        cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
        tr = TaskRecorder(None, cam_motion, fps=30)

        try:
            scene.get_demo(False, callable_each_step=tr.take_snap, randomly_place=False)
        except (WaypointError, NoWaypointsError, DemoError, Exception) as e:
            traceback.print_exc()
            return
        success, terminate = task.success()

        task_result = "success" if success else "failed"

        recording_output_path = f"/tmp/{task_result}/rlbench_video_{task_file}.mp4"

        os.makedirs(os.path.dirname(recording_output_path), exist_ok=True)

        tr.save(recording_output_path, task_description, None)

        print(f"Saving video to {recording_output_path}")

        for object_name in scene_background: 
            Shape(object_name).set_renderable(True)

        if success:
            print("Demo was a success!")


if __name__ == '__main__':
    main()
