from pyrep.objects.object import Object
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper

from abc import ABC
from abc import abstractmethod

import logging

class Robot(ABC):
    """Simple container for the robot components.
    """

    @property
    @abstractmethod
    def is_bimanual(self):
        pass

    @property
    @abstractmethod
    def initial_state(self):
        pass


    @abstractmethod
    def release_gripper(self):
        pass

    @abstractmethod
    def is_in_collision(self):
        pass
    
    @abstractmethod
    def zero_velocity(self):
        pass

    @abstractmethod
    def actutate_gripper(self, amount: float, velocity: float, name: str):
        pass

    @abstractmethod
    def grasp(self, obj: Object, name: str = None):
        pass


class UnimanualRobot(Robot):

    def __init__(self, arm: Arm, gripper: Gripper):
        self.arm = arm
        self.gripper = gripper

    def release_gripper(self):
        self.gripper.release()

    def initial_state(self):
        return [self.arm.get_configuration_tree(), self.gripper.get_configuration_tree()]

    def is_in_collision(self):
        return self.arm.check_arm_collision()

    def zero_velocity(self):
        self.arm.set_joint_target_velocities([0] * len(self.arm.joints))
        self.gripper.set_joint_target_velocities([0] * len(self.gripper.joints))

    def actutate_gripper(self, amount: float, velocity: float, name: str = None):
        return self.gripper.actuate(amount, velocity)

    def grasp(self, obj: Object, name: str = None):
        return self.gripper.grasp(obj)

    @Robot.is_bimanual.getter
    def is_bimanual(self):
        return False
    

class BimanualRobot(Robot):

    def __init__(self, right_arm: Arm, right_gripper: Gripper, left_arm: Arm, left_gripper: Gripper):
        self.right_arm = right_arm
        self.right_gripper = right_gripper
        self.left_arm = left_arm
        self.left_gripper = left_gripper

    def colorize_gripper(self):
        self.right_gripper.colorize([0, 0.8, 0])
        self.left_gripper.colorize([0.8, 0, 0])


    def hide(self, name: str = "both"):
        if "both" in name:
            self.right_arm.hide()
            self.right_gripper.hide()
            self.left_arm.hide()
            self.left_gripper.hide()
        elif "right" in name:
            self.right_arm.hide()
            self.right_gripper.hide()
        elif "left" in name:
            self.left_arm.hide()
            self.left_gripper.hide()

    def release_gripper(self, name: str = 'both'):
        if 'both' in name:
            self.right_gripper.release()
            self.left_gripper.release()
        elif 'right' in name:
            self.right_gripper.release()
        elif 'left' in name:
            self.left_gripper.release()

    def initial_state(self):
        return [(self.right_arm.get_configuration_tree(),
                self.right_gripper.get_configuration_tree()),
                (self.left_arm.get_configuration_tree(),
                self.left_gripper.get_configuration_tree())]

    def is_in_collision(self):
        return self.right_arm.check_arm_collision() or self.left_arm.check_arm_collision()

    def zero_velocity(self):
        self.right_arm.set_joint_target_velocities([0] * len(self.right_arm.joints))
        self.right_gripper.set_joint_target_velocities([0] * len(self.right_gripper.joints))
        self.left_arm.set_joint_target_velocities([0] * len(self.left_arm.joints))
        self.left_gripper.set_joint_target_velocities([0] * len(self.left_gripper.joints))

    def get_arms_by_name(self, name: str):
        if 'right' in name:
            return [self.right_arm]
        if 'left' in name:
            return [self.left_arm]
        if 'both' in name:
            return [self.right_arm, self.left_arm]

    def actutate_gripper(self, amount: float, velocity: float, name: str ='both'):
        logging.debug("actuating gripper for %s", name)
        if 'right' in name:
            return self.right_gripper.actuate(amount, velocity)
        if 'left' in name:
            return self.left_gripper.actuate(amount, velocity)
        if 'both' in name:
            right_done = self.right_gripper.actuate(amount, velocity)
            left_done = self.left_gripper.actuate(amount, velocity)
            return right_done and left_done
        else:
            logging.warning("invalid robot name %s", name)
            return True

    def grasp(self, obj: Object, name: str = None):
        logging.debug("grasping with %s", name)
        if 'right' in name:
            if obj in self.left_gripper.get_grasped_objects():
                logging.warning("Object %s is already grasped by left gripper", obj.get_name())
                return False
            else:
                return self.right_gripper.grasp(obj)
        if 'left' in name:
            if obj in self.right_gripper.get_grasped_objects():
                logging.warning("Object %s is already grasped by right gripper", obj.get_name())
                return False
            else:
                return self.left_gripper.grasp(obj)
        if 'both' in name:
            # object can be only attached to one robot
            if  obj in self.left_gripper.get_grasped_objects():
                right_detected = False
            else:
                right_detected = self.right_gripper.grasp(obj)
            if obj in self.right_gripper.get_grasped_objects():
                left_detected = False
            else:
                left_detected = self.left_gripper.grasp(obj)
            return right_detected or left_detected


    @Robot.is_bimanual.getter
    def is_bimanual(self):
        return True
