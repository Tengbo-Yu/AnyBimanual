from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition

colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
]

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class HandoverItemMedium(BimanualTask):

    def init_task(self) -> None:

        self.items = [Shape(f'item{i}') for i in range(3)]

        for i, (_, color) in enumerate(colors):
            self.items[i].set_color(color)

        self.register_graspable_objects(self.items)

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint5': 'right'})

        self.boundaries = Shape('handover_item_boundary')


    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        color_name, _color = colors[index]


        b = SpawnBoundary([self.boundaries])
        b.clear()
        for item in self.items:
            b.sample(item, min_distance=0.1)
            

        w0 = Dummy('waypoint2')
        w0.set_position([0.0, 0.0, -0.025], relative_to=self.items[index], reset_dynamics=False)
        #w0.set_orientation([-np.pi, 0, -np.pi], relative_to=self.items[index], reset_dynamics=False)

        w1 = Dummy('waypoint1')
        w1.set_position([0.0, 0.0, 0.1], relative_to=self.items[index], reset_dynamics=False)

        w3 = Dummy('waypoint3')
        w3.set_position([0.0, 0.0, 0.3], relative_to=self.items[index], reset_dynamics=False)


        right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        self.register_success_conditions(
            [DetectedCondition(self.items[index], right_success_sensor),  
             DetectedCondition(self.items[index], left_success_sensor, negated=True),
             LiftedCondition(self.items[index], 0.8)])

        return [f'bring me the {color_name} item',
                f'hand over the {color_name} object']

    def variation_count(self) -> int:
        return len(colors)
    
    def boundary_root(self) -> Object:
        return Shape('handover_item_boundary')
    
    def is_static_workspace(self):
        return True

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
