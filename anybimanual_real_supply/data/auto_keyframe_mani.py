"""
Record video for debugging data and observing key points.

Usage:
cd path_to_your_project/data
python auto_keyframe_mani.py
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import natsort
import imageio
import demo_loading_utils

from tqdm import trange
import shutil

# Camera parameter settings
camera_list = ["front_rgb"]

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
    ],
    
}
distortions = {
    'front_rgb': [0.1769687533378601, -0.5447412729263306, -0.0021821269765496254, 0.0002617577847559005, 0.478135347366333],
}

extrinsics = {k: np.array(v) for k, v in extrinsics.items()}
intrinsics = {k: np.array(v) for k, v in intrinsics.items()}
distortions = {k: np.array(v) for k, v in distortions.items()}

# Task name and camera name
task_name = 'pingpang'  # Modify this to your desired task name
camera_name = 'front_rgb'

extrinsic = extrinsics[camera_name]
extrinsic = np.linalg.inv(extrinsic)  # From world coordinates to camera coordinates
intrinsic = intrinsics[camera_name]

# Keyframe selection parameters                                                                                     
keypoint_method = 'heuristic_real'
stopping_delta = 0.0000000
stopping_buffer = 35
warm_up = 60
cool_down = 0

# Set the range of episodes to process
N = 26
start_episode = 0  # Modify this to your desired starting episode index
save_folder = f'data/debug/{task_name}_keypoint_auto'

keyframe_folder_path = os.path.join(save_folder, f'{task_name}_keyframes')

# If the save folder exists, delete and recreate it
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder, exist_ok=True)
os.makedirs(keyframe_folder_path, exist_ok=True)

episode_id_that_should_skip = []

# Set the range of keyframe numbers based on the task name
if task_name == 'press':
    keypoint_number_should_be_min = 1
    keypoint_number_should_be_max = 7
elif task_name == 'handover':
    keypoint_number_should_be_min = 1
    keypoint_number_should_be_max = 7
elif task_name == 'lift':
    keypoint_number_should_be_min = 1
    keypoint_number_should_be_max = 5
elif task_name == 'pick_in_two':
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 6
elif task_name == 'pick_in_one':
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 6
elif task_name == 'pour':
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 6
elif task_name == 'pingpang':
    keypoint_number_should_be_min = 1
    keypoint_number_should_be_max = 6
elif task_name == 'clothes':
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 6

for episode_id in trange(start_episode, N):
    episode_path = f'data/real_dual_ur_train_data/{task_name}/all_variations/episodes/episode{episode_id}/'
    if not os.path.exists(episode_path):
        print(f"Skipping episode {episode_id}: Path does not exist")
        continue
    description_path = os.path.join(episode_path, 'variation_descriptions.pkl')
    description = pickle.load(open(description_path, 'rb'))
    description = description[0]

    # Get observation data and list of image files
    rgb_path = os.path.join(episode_path, camera_name)
    demo_path = os.path.join(episode_path, 'low_dim_obs.pkl')

    img_files = os.listdir(rgb_path)
    img_files = natsort.natsorted(img_files)

    demo = pickle.load(open(demo_path, 'rb'))

    # Automatically select keyframes using the keyframe selection function
    episode_keypoints = demo_loading_utils._keypoint_discovery_dualarm1101(
        demo,
        warm_up=warm_up,
        cool_down=cool_down,
        include_last_frame=False,
    )

    # Add the first frame as a keyframe by default
    first_img_file = img_files[0]
    first_step = int(os.path.splitext(first_img_file)[0])
    if first_step not in episode_keypoints:
        episode_keypoints.insert(0, first_step)

    # Ensure the keyframe list is sorted in chronological order
    episode_keypoints = sorted(set(episode_keypoints))

    # Check if the number of keyframes is within the specified range
    keypoint_num = len(episode_keypoints)
    if keypoint_num < keypoint_number_should_be_min or keypoint_num > keypoint_number_should_be_max:
        episode_id_that_should_skip.append(episode_id)
        print(f"Episode {episode_id} skipped due to keypoint number {keypoint_num}")
        continue

    # Save video
    save_path = f'{save_folder}/episode{episode_id}_keynum_{keypoint_num}_automatic.mp4'
    video_writer = imageio.get_writer(save_path, fps=20)

    for img_file in img_files:
        img_path = os.path.join(rgb_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the observation data corresponding to the current frame
        step = int(os.path.splitext(img_file)[0])

        vel_left = demo[step].left.joint_velocities
        vel_right = demo[step].right.joint_velocities
        vel_left = np.max(np.abs(vel_left))
        vel_right = np.max(np.abs(vel_right))

        gripper_openness_right = demo[step].right.gripper_open
        gripper_openness_left = demo[step].left.gripper_open

        # Draw the position of the right gripper in the image
        pos_world_right = demo[step].right.gripper_pose[:3]
        pos_world_right = np.array(pos_world_right).reshape(3, 1)
        pos_cam_right = extrinsic[:3, :3] @ pos_world_right + extrinsic[:3, 3:]
        pos_cam_right = intrinsic @ pos_cam_right
        pos_cam_right = pos_cam_right / pos_cam_right[2]
        pos_cam_right = pos_cam_right[:2].flatten()
        pos_cam_right = pos_cam_right.astype(np.int32)

        # Draw the position of the left gripper in the image
        pos_world_left = demo[step].left.gripper_pose[:3]
        pos_world_left = np.array(pos_world_left).reshape(3, 1)
        pos_cam_left = extrinsic[:3, :3] @ pos_world_left + extrinsic[:3, 3:]
        pos_cam_left = intrinsic @ pos_cam_left
        pos_cam_left = pos_cam_left / pos_cam_left[2]
        pos_cam_left = pos_cam_left[:2].flatten()
        pos_cam_left = pos_cam_left.astype(np.int32)

        # Draw on the image
        color_right = (255, 0, 0)  # Red for right gripper
        color_left = (0, 0, 255)   # Blue for left gripper
        cv2.circle(img, tuple(pos_cam_right), 5, color_right, -1)
        cv2.circle(img, tuple(pos_cam_left), 5, color_left, -1)

        # Display some information
        if step in episode_keypoints:
            color = (0, 0, 255)  # Red font for keyframes
        else:
            color = (0, 255, 0)  # Green font

        cv2.putText(img, f"step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"right_velocity: {vel_right:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"left_velocity: {vel_left:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"right_gripper_openness: {gripper_openness_right}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"left_gripper_openness: {gripper_openness_left}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save video frame
        video_writer.append_data(img)

    video_writer.close()
    print(f"Video saved to {save_path}")

    # Save keyframe data (rgb, depth, low_dim_obs.pkl, variation_descriptions.pkl, variation_number.pkl)
    keyframe_folder = os.path.join(keyframe_folder_path, f'episode{episode_id}')
    os.makedirs(keyframe_folder, exist_ok=True)

    # Save variation_descriptions.pkl and variation_number.pkl
    shutil.copy(os.path.join(episode_path, 'variation_descriptions.pkl'), keyframe_folder)
    shutil.copy(os.path.join(episode_path, 'variation_number.pkl'), keyframe_folder)

    # Initialize an empty list to save keyframe observations
    keyframe_demos = []

    # Create a mapping from step to image index
    step_to_img_idx = {}
    for idx_img, img_file in enumerate(img_files):
        step_img = int(os.path.splitext(img_file)[0])
        step_to_img_idx[step_img] = idx_img

    # Create destination folders
    dst_img_folder = os.path.join(keyframe_folder, 'front_rgb')
    dst_depth_folder = os.path.join(keyframe_folder, 'front_depth')
    dst_nerf_folder = os.path.join(keyframe_folder, 'nerf_data')
    os.makedirs(dst_img_folder, exist_ok=True)
    os.makedirs(dst_depth_folder, exist_ok=True)
    os.makedirs(dst_nerf_folder, exist_ok=True)

    # Save keyframe rgb and depth images and corresponding observation data
    for idx, step in enumerate(episode_keypoints):
        img_filename = f'{step}.png'
        depth_filename = f'{step}.png'  # Depth image filename is the same as RGB

        # Source file paths
        src_img_path = os.path.join(episode_path, 'front_rgb', img_filename)
        src_depth_path = os.path.join(episode_path, 'front_depth', depth_filename)

        # Destination filenames, named in order 0, 1, 2, 3
        dst_img_filename = f'{idx}.png'
        dst_depth_filename = f'{idx}.png'

        # Destination file paths
        dst_img_path = os.path.join(dst_img_folder, dst_img_filename)
        dst_depth_path = os.path.join(dst_depth_folder, dst_depth_filename)

        # Copy files
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_depth_path, dst_depth_path)

        # Save corresponding observation data to keyframe_demos list
        keyframe_demos.append(demo[step])

        # Copy corresponding nerf_data subfolders
        idx_in_img_files = step_to_img_idx[step]
        src_nerf_subfolder = os.path.join(episode_path, 'nerf_data', str(idx_in_img_files))
        dst_nerf_subfolder = os.path.join(dst_nerf_folder, str(idx))
        shutil.copytree(src_nerf_subfolder, dst_nerf_subfolder)

    # After processing all keyframes, save keyframe_demos to low_dim_obs.pkl
    with open(os.path.join(keyframe_folder, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(keyframe_demos, f)

print("All processing completed.")
