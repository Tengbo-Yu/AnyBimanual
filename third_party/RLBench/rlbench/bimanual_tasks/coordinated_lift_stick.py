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
from rlbench.backend.conditions import Condition
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class CoordinatedLiftStick(BimanualTask):

    def init_task(self) -> None:

        self.queue = Shape('queue')
        self.register_graspable_objects([self.queue])

        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint2': 'left',  'waypoint4': 'left',  'waypoint6': 'left'})


    def init_episode(self, index) -> List[str]:
        self._variation_index = index

        self.register_success_conditions([LiftedCondition(self.queue, 1.2),])
        return ['lift the stick']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]