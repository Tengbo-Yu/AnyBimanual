from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from rlbench.backend.exceptions import BoundaryError
from absl import logging
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition


colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('yellow', (1.0, 1.0, 0.0)),
    #('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    #('teal', (0, 0.5, 0.5)),
    #('black', (0.0, 0.0, 0.0)),
    #('white', (1.0, 1.0, 1.0)),
]

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class HandoverItem(BimanualTask):

    def init_task(self) -> None:

        self.items = [Shape(f'item{i}') for i in range(5)]

        self.register_graspable_objects(self.items)

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint5': 'right'})

        self.boundaries = Shape('handover_item_boundary')
        self.boundary_init_pose = Shape('handover_item_boundary').get_position()


    def init_episode(self, index:  int) -> List[str]:

        self._variation_index = index

        color_name, color = colors[index]
        self.items[0].set_color(color)

        remaining_colors = colors.copy()
        remaining_colors.remove((color_name, color))
        np.random.shuffle(remaining_colors)

        for i, item in enumerate(self.items[1:]):
            item.set_color(remaining_colors[i][1])

        Shape('handover_item_boundary').set_position(self.boundary_init_pose)

        b = SpawnBoundary([self.boundaries])
        b.MAX_SAMPLES = 1000
        b.clear()
        for item in self.items:
            try:
                b.sample(item, min_distance=0.05)
            except BoundaryError as err:
                logging.warning("error %s. Sampling again while ignoring collisions ", err)
                b.sample(item, ignore_collisions=True, min_distance=0.05)

        right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        self.register_success_conditions(
            [DetectedCondition(self.items[0], right_success_sensor),  
             DetectedCondition(self.items[0], left_success_sensor, negated=True),
             LiftedCondition(self.items[0], 0.8)])

        return [f'bring me the {color_name} item',
                f'hand over the {color_name} object']

    def variation_count(self) -> int:
        return len(colors)
    
    #def boundary_root(self) -> Object:
    #    return Shape('handover_item_boundary')

    def is_static_workspace(self):
        return True

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
