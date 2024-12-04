import numpy as np
from os import path

from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.const import TextureMappingMode
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.const import colors

ASSET_DIR = path.join(path.dirname(path.abspath(__file__)), '../../', 'tests', 'unit', 'assets', 'textures')
NUM_TEXTURES = 10

TEX_KWARGS = {
    'mapping_mode': TextureMappingMode.PLANE,
    'repeat_along_u': True,
    'repeat_along_v': True,
}


class OpenDrawerTestTextureFrame(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')

    def init_episode(self, index: int) -> List[str]:
        texture_idx = np.random.randint(NUM_TEXTURES)
        texture_file = path.join(ASSET_DIR, '%d.png' % texture_idx)
        text_ob, texture = self.pyrep.create_texture(texture_file)

        drawer = Shape('drawer')
        drawer0 = Shape('drawer0')
        drawer1 = Shape('drawer1')
        drawer2 = Shape('drawer2')
        drawer_frame = Shape('drawer_frame')

        drawer.set_texture(texture, **TEX_KWARGS)
        drawer0.set_texture(texture, **TEX_KWARGS)
        drawer1.set_texture(texture, **TEX_KWARGS)
        drawer2.set_texture(texture, **TEX_KWARGS)
        drawer_frame.set_texture(texture, **TEX_KWARGS)

        # ungrouped = drawer_frame.ungroup()
        # for o in ungrouped:
        #     o.set_texture(texture, **TEX_KWARGS)
        # self.pyrep.group_objects(ungrouped)
        
        # drawer_frame.set_texture(texture, **TEX_KWARGS)
        text_ob.remove()
        self.pyrep.step()

        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.15)])

        return ['open the %s drawer' % (option),
                'grip the %s handle and pull the %s drawer open' % (
                    option, option),
                'slide the %s drawer open' % (option)]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
