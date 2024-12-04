from pyrep.robots.arms.arm import Arm


class PandaRight(Arm):

    def __init__(self, count: int = 0):
        # right arm
        super().__init__(count, 'Panda_rightArm', 7, base_name='DualPanda')


class PandaLeft(Arm):

    def __init__(self, count: int = 0):
        # left arm
        super().__init__(count, 'Panda_leftArm', 7, base_name='DualPanda')