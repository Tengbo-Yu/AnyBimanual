"""
Convert NTU collected data into RLBench format.

Usage:
cd path_to_your_project/data
python preprocess_ntu_dualarm.py
"""

import sys
import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from termcolor import cprint
from tqdm import trange
import ast
from rlbench.backend.observation import Observation, UnimanualObservationData, BimanualObservation
from rlbench.observation_config import ObservationConfig
from rlbench.demo import Demo
from rlbench.backend.const import *
from rlbench.backend.utils import float_array_to_grayscale_image, float_array_to_rgb_image
from scipy.spatial.transform import Rotation
from natsort import natsorted
import shutil


def save_extrinsic_and_intrinsic(path, extrinsic, intrinsic):
    """Save extrinsic and intrinsic camera parameters to a file."""
    with open(path, 'w') as f:
        for row in extrinsic:
            for ele in row:
                f.write('{:.6f}'.format(ele) + ' ')
            f.write('\n')
        f.write('\n')
        for row in intrinsic:
            for ele in row:
                f.write('{:.6f}'.format(ele) + ' ')
            f.write('\n')


def read_txt_to_matrix(filename):
    """Read robot.txt and convert to a matrix."""
    with open(filename, 'r') as file:
        # Read each line of the file
        lines = file.read().strip().replace('\n', '')
        lines = '[' + lines + ']'
        # Example for debugging:
        # lines = "[[2, 2, 2], [3, 3, 4], [4, 5, 1], [6, 6, 1]]"
        matrix = ast.literal_eval(lines)
    return matrix


def parse_camera_info(filename):
    """Parse camera intrinsic parameters from a file."""
    with open(filename, 'r') as file:
        camera_info = file.read().strip()
    K = np.zeros((3, 3), dtype=np.float64)
    K[2, 2] = 1.0
    distortion = np.zeros((1, 5), dtype=np.float64)
    lines = camera_info.split('\n')
    for line in lines:
        if 'Principal Point' in line:
            principal_point = line.split(':')[-1].strip().split(',')
            K[0, 2] = float(principal_point[0])
            K[1, 2] = float(principal_point[1])
        elif 'Focal Length' in line:
            focal_length = line.split(':')[-1].strip().split(',')
            K[0, 0] = float(focal_length[0])
            K[1, 1] = float(focal_length[1])
        elif 'Distortion Coefficients' in line:
            distortion_coefficients = line.split(':')[-1].strip()
            distortion_coefficients = distortion_coefficients.replace('[', '').replace(']', '')
            distortion_coefficients = list(map(float, distortion_coefficients.split(',')))
            distortion[0, :len(distortion_coefficients)] = distortion_coefficients

    # Convert mm to m if necessary
    # K = K / 1000.0

    return K  # Almost no distortion


instruction = "Pour the ping pong ball from one cup into another."  # Instruction from peract1


def main():
    current = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current)

    root_path = 'path_to_raw_data'  # Replace with your raw data path
    save_path = 'path_to_save_converted_data'  # Replace with your desired save path

    # Note: x-left, y-backward, z-up; extrinsic is camera to world
    camera_list = ['camera1']
    new_camera_map = {
        camera_list[0]: 'front_rgb',
    }
    extrinsics = {
        camera_list[0]: [
            [0.9984647, -0.01066009, -0.05435628, 0.41119387],
            [0.05308204, -0.09627122, 0.9939387, -1.28711702],
            [-0.01582842, -0.99529805, -0.09555755, 0.33819046],
            [0.0, 0.0, 0.0, 1.0]
        ],
    }
    intrinsics = {
        camera_list[0]: [
            [603.2314453125, 0., 325.3480529785156],
            [0., 603.2608032226562, 251.1649932861328],
            [0, 0, 1],
        ],  # Color camera intrinsics
    }
    distortions = {
        camera_list[0]: [0.1769687533378601, -0.5447412729263306,
                         -0.0021821269765496254, 0.0002617577847559005,
                         0.478135347366333],
    }
    extrinsics = {k: np.array(v) for k, v in extrinsics.items()}
    intrinsics = {k: np.array(v) for k, v in intrinsics.items()}
    distortions = {k: np.array(v) for k, v in distortions.items()}

    old_episode_folders = natsorted(os.listdir(root_path))
    eids = len(old_episode_folders)

    skip_list = []

    description_map = {k: instruction for k in range(eids)}
    variation_map = {k: 0 for k in range(len(description_map))}

    if os.path.exists(save_path):
        # Ask the user if they want to overwrite the existing data
        cprint('The folder already exists. Do you want to overwrite the existing data? (y/n)', 'yellow')
        response = input()
        if response.lower() == 'y':
            shutil.rmtree(save_path)
        else:
            cprint('Exiting...', 'red')
            sys.exit()

    fps = 10
    skip_interval = 1
    real_fps = fps // skip_interval

    traj_len_list = []
    left_gripper_pose_list = []
    right_gripper_pose_list = []
    left_vel_list = []
    right_vel_list = []

    for eid in trange(eids):
        if eid in skip_list:
            continue

        gripper_pose_list_tmp = []

        gripper_path = os.path.join(root_path, old_episode_folders[eid], 'gripper.txt')
        traj_path = os.path.join(root_path, old_episode_folders[eid], 'traj.txt')

        with open(traj_path, 'r') as f:
            traj_raw = f.readlines()
        with open(gripper_path, 'r') as f:
            gripper_raw = f.readlines()

        id_list = range(0, len(gripper_raw), skip_interval)
        episode_folder = 'episode' + str(eid)

        # Create the episode folder if it doesn't exist
        os.makedirs(os.path.join(save_path, episode_folder), exist_ok=True)

        demo = []
        # Save images, depth, and nerf data
        for new_id, id in enumerate(id_list):
            img_file = f'_frames/frame{id:06d}.jpg'
            depth_file = f'_depths/depth{id:06d}.png'

            for camera in camera_list:
                new_camera = new_camera_map[camera]

                data_path = os.path.join(root_path, old_episode_folders[eid], camera)

                img = cv2.imread(data_path + img_file, cv2.IMREAD_UNCHANGED)  # (480, 640, 3), 0-255
                depth = cv2.imread(data_path + depth_file, cv2.IMREAD_UNCHANGED)  # (480, 640)

                depth = depth / 1000.0  # Convert depth to meters

                DEBUG = False
                if DEBUG:
                    # Show depth image
                    depth_vis = depth.copy()
                    depth_vis = (depth_vis - np.min(depth_vis)) / (np.max(depth_vis) - np.min(depth_vis))  # Normalize
                    plt.figure()
                    plt.imshow(depth_vis)
                    plt.colorbar()
                    plt.show()
                    plt.savefig(f'depth_{new_id}.png')
                    plt.close()

                depth = float_array_to_rgb_image(depth, scale_factor=DEPTH_SCALE)

                # Save images and depth
                camera_path = os.path.join(save_path, episode_folder, new_camera)
                os.makedirs(camera_path, exist_ok=True)

                if new_camera == 'nerf_data':
                    img_folder = os.path.join(save_path, episode_folder, new_camera, str(new_id))
                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)
                        os.makedirs(os.path.join(img_folder, 'images'))
                        os.makedirs(os.path.join(img_folder, 'depths'))
                        os.makedirs(os.path.join(img_folder, 'poses'))
                    # Save image
                    img_path = os.path.join(img_folder, 'images', str(0) + '.png')
                    cv2.imwrite(img_path, img)

                    depth_path = img_path.replace('images', 'depths')
                    depth.save(depth_path)
                    pose_path = img_path.replace('images', 'poses').replace('.png', '.txt')
                    transformation_matrix = extrinsics[camera]
                    intrinsic_matrix = intrinsics[camera]
                    save_extrinsic_and_intrinsic(pose_path, transformation_matrix, intrinsic_matrix)
                else:
                    assert new_camera == 'front_rgb'  # Should be 'front_rgb'
                    os.makedirs(camera_path.replace('front_rgb', 'front_depth'), exist_ok=True)

                    # Save image
                    img_path = os.path.join(save_path, episode_folder, new_camera, str(new_id) + '.png')
                    cv2.imwrite(img_path, img)

                    # Save depth
                    depth_path = img_path.replace('front_rgb', 'front_depth')
                    depth.save(depth_path)

            # Assuming each line of data is in the format "right_arm_data,left_arm_data"
            # Split each line and process left and right arms separately
            arms_data = traj_raw[id].strip().split(',')
            right_traj_data = arms_data[:16]  # Each arm now has 16 data items
            left_traj_data = arms_data[16:]

            # Convert string data to float and reshape into 4x4 matrices
            right_traj = np.array(right_traj_data, dtype=np.float32).reshape(4, 4)
            left_traj = np.array(left_traj_data, dtype=np.float32).reshape(4, 4)

            # Extract rotation matrices and translation vectors
            right_rotation = right_traj[:3, :3]
            left_rotation = left_traj[:3, :3]
            right_translation = right_traj[:3, 3]
            left_translation = left_traj[:3, 3]

            # Get quaternion representation from rotation matrices
            right_rotation = Rotation.from_matrix(right_rotation)
            left_rotation = Rotation.from_matrix(left_rotation)
            right_quat = right_rotation.as_quat()
            left_quat = left_rotation.as_quat()

            misc = {
                'descriptions': [instruction],
                'front_camera_extrinsics': extrinsics[camera_list[0]],
                'front_camera_intrinsics': intrinsics[camera_list[0]],
                'front_camera_near': 0.0,
                'front_camera_far': 1.2,
            }

            # Get corresponding gripper position data
            right_gripper_joint_position = float(gripper_raw[id].strip().split(',')[0])  # Assuming right arm is first
            left_gripper_joint_position = float(gripper_raw[id].strip().split(',')[1])   # Assuming left arm is second

            # Construct gripper pose lists
            right_gripper_pose = [right_translation[0], right_translation[1], right_translation[2], *right_quat]
            left_gripper_pose = [
                left_translation[0] + 0.78,  # Add a translation of 0.78 meters along the x-axis
                left_translation[1],
                left_translation[2],
                *left_quat
            ]

            right_gripper_pose_list.append(right_gripper_pose)
            left_gripper_pose_list.append(left_gripper_pose)

            # Calculate velocities (example considers position only)
            if new_id == 0:
                right_vel = np.zeros(3)
                left_vel = np.zeros(3)
            else:
                right_vel = (np.array(right_gripper_pose[:3]) - np.array(right_gripper_pose_list[-2][:3])) * real_fps
                left_vel = (np.array(left_gripper_pose[:3]) - np.array(left_gripper_pose_list[-2][:3])) * real_fps

            right_vel_list.append(right_vel)
            left_vel_list.append(left_vel)

            obs = BimanualObservation(
                perception_data={
                    'left_shoulder_rgb': None,  # Fill in as per your application
                    'left_shoulder_depth': None,
                    'right_shoulder_rgb': None,
                    'right_shoulder_depth': None,
                    'overhead_rgb': None,
                    'overhead_depth': None,
                    'wrist_rgb': None,
                    'wrist_depth': None,
                    'front_rgb': None,
                    'front_depth': None,
                },
                task_low_dim_state=None,  # Add task-related low-dimensional state data if available
                misc=misc,
                right=UnimanualObservationData(
                    joint_velocities=right_vel,
                    joint_positions=None,  # Fill or compute actual joint positions
                    joint_forces=None,  # Fill or compute actual joint forces
                    gripper_open=(1.0 if right_gripper_joint_position < 100 else 0.0),
                    gripper_pose=right_gripper_pose,
                    gripper_matrix=None,  # Fill if actual gripper matrix data is available
                    gripper_joint_positions=np.array([right_gripper_joint_position, right_gripper_joint_position]),
                    gripper_touch_forces=None,  # Assume data
                    ignore_collisions=True,  # Assume data
                ),
                left=UnimanualObservationData(
                    joint_velocities=left_vel,
                    joint_positions=None,  # Fill or compute actual joint positions
                    joint_forces=None,  # Fill or compute actual joint forces
                    gripper_open=(1.0 if left_gripper_joint_position < 100 else 0.0),
                    gripper_pose=left_gripper_pose,
                    gripper_matrix=None,  # Fill if actual gripper matrix data is available
                    gripper_joint_positions=np.array([left_gripper_joint_position, left_gripper_joint_position]),
                    gripper_touch_forces=None,  # Assume data
                    ignore_collisions=True,  # Assume data
                )
            )
            demo.append(obs)

        traj_len_list.append(len(demo))
        demo = Demo(demo)
        with open(os.path.join(os.path.join(save_path, episode_folder), LOW_DIM_PICKLE), 'wb') as f:
            pickle.dump(demo, f)
        # Save descriptions
        with open(os.path.join(os.path.join(save_path, episode_folder), VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump([instruction], f)
        # Save variation number
        with open(os.path.join(os.path.join(save_path, episode_folder), VARIATION_NUMBER), 'wb') as f:
            pickle.dump(variation_map[eid], f)


if __name__ == '__main__':
    main()
