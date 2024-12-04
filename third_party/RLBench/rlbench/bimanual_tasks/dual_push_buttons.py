from typing import List
import itertools
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import JointCondition, ConditionSet
from rlbench.backend.task import BimanualTask
from collections import defaultdict

MAX_VARIATIONS = 50

# button top plate and wrapper will be be red before task completion
# and be changed to cyan upon success of task, so colors list used to randomly vary colors of
# base block will be redefined, excluding red and green
colors = [
    ('maroon', (0.5, 0.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]

#color_permutations = list(itertools.permutations(colors, 3))


def print_permutations(color_permutations):
    # pretty printing color_permutations for debug
    print('num permutations: ', str(len(color_permutations)))
    print('color_permutations:\n')
    for i in range(len(color_permutations)):
        print(str(color_permutations[i]))
        if ((i + 1) % 16 == 0): print('')


class DualPushButtons(BimanualTask):



    def init_task(self) -> None:
        self.buttons_pushed = 0
        self.color_variation_index = 0
        self.target_buttons = [Shape('push_buttons_target%d' % i)
                               for i in range(3)]
        self.target_topPlates = [Shape('target_button_topPlate%d' % i)
                                 for i in range(3)]
        self.target_joints = [Joint('target_button_joint%d' % i)
                              for i in range(3)]
        self.target_wraps = [Shape('target_button_wrap%d' % i)
                             for i in range(3)]
        self.boundaries = Shape('push_buttons_boundary')
        # goal_conditions merely state joint conditions for push action for
        # each button regardless of whether the task involves pushing it
        self.goal_conditions = [JointCondition(self.target_joints[n], 0.001)
                                for n in range(2)]

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint2': 'right'})

    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        for tp in self.target_topPlates:
            tp.set_color([1.0, 0.0, 0.0])
        for w in self.target_wraps:
            w.set_color([1.0, 0.0, 0.0])
            
        #button_colors = color_permutations[index]

        rng = np.random.default_rng(index)
        button_colors = rng.choice(np.asarray(colors, dtype=object), 3, replace=False)
        
        self.color_names = []
        self.color_rgbs = []
        self.chosen_colors = []

        for i, b in enumerate(self.target_buttons):
            color_name, color_rgb = button_colors[i]
            self.color_names.append(color_name)
            self.color_rgbs.append(color_rgb)
            self.chosen_colors.append((color_name, color_rgb))
            b.set_color(color_rgb)

        self.register_success_conditions([ConditionSet(self.goal_conditions, True, True)])

        # ..todo separate the spawn boundaries for the left and right  robot


        right_boundary = Shape('push_buttons_boundary_right')
        b = SpawnBoundary([right_boundary])
        b.sample(self.target_buttons[0], min_distance=0.1)

        left_boundary = Shape('push_buttons_boundary_left')
        b = SpawnBoundary([left_boundary])
        b.sample(self.target_buttons[1], min_distance=0.1)

        b = SpawnBoundary([self.boundaries])
        b.sample(self.target_buttons[2], min_distance=0.1)


        w0 = Dummy('waypoint0')
        x, y, z = self.target_buttons[0].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

        w0 = Dummy('waypoint1')
        x, y, z = self.target_buttons[1].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

        rtn0 = f'push the {self.color_names[0]} and the {self.color_names[1]} buttons'
        rtn1 = f'press the {self.color_names[0]} and the {self.color_names[1]} buttons'
        rtn2 = f'push down the buttons with the {self.color_names[0]} and the the {self.color_names[1]} base'


        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return MAX_VARIATIONS

    def cleanup(self) -> None:
        self.buttons_pushed = 0

    def base_rotation_bounds(self):
        return [0.0] * 3, [0.0] * 3

    def is_static_workspace(self):
        return True

    def _move_above_next_target(self, waypoint):
        if self.buttons_pushed >= self.buttons_to_push:
            print('buttons_pushed:', self.buttons_pushed, 'buttons_to_push:',
                  self.buttons_to_push)
            raise RuntimeError('Should not be here.')
