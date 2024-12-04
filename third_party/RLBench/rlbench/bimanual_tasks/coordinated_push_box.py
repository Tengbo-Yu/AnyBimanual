from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask

class CoordinatedPushBox(BimanualTask):

    def init_task(self) -> None:

        self.item = Shape('cube')
        self.target = Shape('target')
        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint2': 'right'})


    def init_episode(self, index) -> List[str]:
        self._variation_index = index

        #Dummy('waypoint0').set_position(position=(0,0,0),relative_to=self.target)
        #Dummy('waypoint0').set_position(position=(0,0,0),relative_to=self.target)

        success_sensor = ProximitySensor('success0')
        self.register_success_conditions([DetectedCondition(self.item, success_sensor)])
        return ['push the box to the red area']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        angle = np.deg2rad(10)
        return [0, 0, - angle], [0, 0, angle]
