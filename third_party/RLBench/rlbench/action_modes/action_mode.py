from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode
from rlbench.action_modes.arm_action_modes import BimanualJointPosition, JointPosition
from rlbench.action_modes.gripper_action_modes import GripperActionMode
from rlbench.action_modes.gripper_action_modes import BimanualGripperJointPosition, GripperJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError('You must define your own action bounds.')




class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])
        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))



class BimanualMoveArmThenGripper(MoveArmThenGripper):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene: Scene, action: np.ndarray):

        assert(len(action) == 18)

        arm_action_size = np.prod(self.arm_action_mode.unimanual_action_shape(scene))
        ee_action_size = np.prod(self.gripper_action_mode.unimanual_action_shape(scene))
        ignore_collisions_size = 1

        action_size = arm_action_size + ee_action_size + ignore_collisions_size

        assert(action_size == 9)

        right_action = action[:action_size]
        left_action = action[action_size:]

        right_arm_action = np.array(right_action[:arm_action_size])
        left_arm_action = np.array(left_action[:arm_action_size])

        arm_action = np.concatenate([right_arm_action, left_arm_action], axis=0)        

        right_ee_action = np.array(right_action[arm_action_size:arm_action_size+ee_action_size])
        left_ee_action = np.array(left_action[arm_action_size:arm_action_size+ee_action_size])
        ee_action = np.concatenate([right_ee_action, left_ee_action], axis=0)

        right_ignore_collisions = bool(right_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        left_ignore_collisions = bool(left_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        ignore_collisions = [right_ignore_collisions, left_ignore_collisions]

        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)


    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)) + 2


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])



class BimanualJointPositionActionMode(ActionMode):

    def __init__(self, arm_action_mode=None, gripper_action_mode=None):
        arm_action_mode = arm_action_mode or BimanualJointPosition()
        gripper_action_mode = gripper_action_mode or BimanualDiscrete()

        super(BimanualJointPositionActionMode, self).__init__(arm_action_mode, gripper_action_mode)

    def action(self, scene: Scene, action: np.ndarray):

        assert(action.shape == (16,))

        
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        assert(arm_act_size == 14)

        arm_action = np.concatenate([action[0:7], action[8:15]], axis=0 )
        ee_action = np.array([action[7], action[15]])


        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)

        self.arm_action_mode.action_step(scene)

        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise Exception("Not implemented yet.")
