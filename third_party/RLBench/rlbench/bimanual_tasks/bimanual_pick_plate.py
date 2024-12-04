from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from collections import defaultdict


from rlbench.backend.conditions import Condition

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False



class BimanualPickPlate(BimanualTask):

    def init_task(self) -> None:
        self.plate = Shape('plate')
      
        self.register_success_conditions([ LiftedCondition(self.plate, 1.0)])
        self.register_graspable_objects([self.plate])
        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint2': 'left', 'waypoint6': 'left'})

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the plate']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
