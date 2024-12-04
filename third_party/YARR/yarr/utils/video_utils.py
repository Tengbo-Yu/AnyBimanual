import os
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class TaskRecorder(object):

    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        self._current_snaps.append(
            (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))

    def save(self, path, lang_goal, reward):
        print(f"Converting to video ... {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2
        image_size = self._cam_motion.cam.get_resolution()
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
                tuple(image_size))
        
        for image in self._current_snaps:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = (0.45 * image_size[0]) / 640
            font_thickness = 2


            if lang_goal:

                lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
                lang_textX = (image_size[0] - lang_textsize[0]) // 2

                frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
                                    fontScale=font_scale, fontFace=font, color=(0, 0, 0),
                                    thickness=font_thickness, lineType=cv2.LINE_AA)
                video.write(frame)
        video.release()
        self._current_snaps = []
