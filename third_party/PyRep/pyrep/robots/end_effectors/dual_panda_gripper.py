from pyrep.robots.end_effectors.gripper import Gripper



class PandaGripperRight(Gripper):

    def __init__(self, count: int = 0):
        super().__init__(count, 'Panda_rightArm_gripper',
                         ['Panda_rightArm_gripper_joint1', 'Panda_rightArm_gripper_joint2'])



class PandaGripperLeft(Gripper):

    def __init__(self, count: int = 0):
        super().__init__(count, 'Panda_leftArm_gripper',
                         ['Panda_leftArm_gripper_joint1', 'Panda_leftArm_gripper_joint2'])





