from abc import abstractmethod
import numpy as np
from typing import List
from typing import Dict
from typing import Any

from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class ImageType(Enum):
    RGB = 1
    DEPTH = 2
    MASK = 3

@dataclass
class CameraImage:
    
    name: str
    dtype: ImageType
    data: np.ndarray

@dataclass
class Observation:
    """Storage for both visual and low-dimensional observations."""

    #..todo:: replace

    perception_data: Dict[str, np.ndarray]
  
    task_low_dim_state: np.ndarray
    
    misc: Dict[str, Any]

    @property
    @abstractmethod
    def is_bimanual(self):
        pass


#..todo:: rename to ProprioceptionObservation

@dataclass 
class UnimanualObservationData:

    joint_velocities: np.ndarray
    joint_positions: np.ndarray
    joint_forces: np.ndarray
    gripper_open: float
    gripper_pose: np.ndarray
    gripper_matrix: np.ndarray
    gripper_joint_positions: np.ndarray
    gripper_touch_forces: np.ndarray

    ignore_collisions: np.ndarray

@dataclass
class UnimanualObservation(UnimanualObservationData, Observation):
    
    
    @Observation.is_bimanual.getter
    def is_bimanual(self):
        return False

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])


@dataclass
class BimanualObservation(Observation):
    
    right: UnimanualObservationData = None
    left: UnimanualObservationData = None

    @Observation.is_bimanual.getter
    def is_bimanual(self):
        return True

    def get_low_dim_data(self, robot: UnimanualObservationData) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if robot.gripper_open is None else [[robot.gripper_open]]
        #for data in [robot.joint_velocities, robot.joint_positions,
        #             robot.joint_forces,
        #             robot.gripper_pose, robot.gripper_joint_positions,
        #             robot.gripper_touch_forces, self.task_low_dim_state]:

        for data in [robot.gripper_joint_positions]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
