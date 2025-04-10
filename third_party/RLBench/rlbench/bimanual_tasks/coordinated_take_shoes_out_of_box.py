from typing import List, Tuple
import numpy as np
from collections import defaultdict
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.task import BimanualTask

class CoordinatedTakeShoesOutOfBox(BimanualTask):

    def init_task(self) -> None:
        shoe1, shoe2 = Shape('shoe1'), Shape('shoe2')
        self.register_graspable_objects([shoe1, shoe2])
        success_sensor = ProximitySensor('success_out_box')
        self.register_success_conditions([
            DetectedCondition(shoe1, success_sensor),
            DetectedCondition(shoe2, success_sensor),
            NothingGrasped(self.robot.right_gripper),
            NothingGrasped(self.robot.left_gripper)])
        
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(4):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['take shoes out of box',
                'open the shoe box and take the shoes out',
                'put the shoes found inside the box on the table',
                'set the shoes down on the table',
                'pick up the shoes from the box and put them down',
                'grasp the edge of the box lid to open it, then grasp each shoe'
                ', lifting up out of the shoe box and leaving them down on the '
                'table']

    def variation_count(self) -> int:
        return 1
    
    def is_static_workspace(self):
        return True
    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]
