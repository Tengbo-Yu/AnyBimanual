"""
Control the UR robot arm with the learned policy.

Usage:
conda activate your_env
CUDA_VISIBLE_DEVICES=0 python eval_agent_on_robot.py
"""

import os
import time
from PIL import Image
import numpy as np
import pickle
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from real_utils import image_to_float_array, float_array_to_rgb_image, pointcloud_from_depth_and_camera_params
DEPTH_SCALE = 2**24 - 1

from agents import any_bimanual

from scipy.spatial.transform import Rotation as R
from helpers import demo_loading_utils
from helpers.clip.core.clip import tokenize

import visdom
import einops
# from transformers import CLIPTokenizer, CLIPTextModel

class AnyBimanualAgentInterface:
    def __init__(self, cfg, instruction):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step = 0

        current_dir = os.path.dirname(os.path.realpath(__file__))
        print('Current directory:', current_dir)

        # Camera extrinsic parameters (pre-calibrated)
        self.extrinsics = {
            'front_rgb': [
                [0.9984647, -0.01066009, -0.05435628, 0.41119387],
                [0.05308204, -0.09627122, 0.9939387, -1.28711702],
                [-0.01582842, -0.99529805, -0.09555755, 0.33819046],
                [0.0, 0.0, 0.0, 1.0]
            ],
        }
        self.extrinsics = {k: np.array(v) for k, v in self.extrinsics.items()}

        # Camera intrinsic parameters
        self.intrinsics = {
            'front_rgb': [
                [603.2314453125, 0., 325.3480529785156],
                [0., 603.2608032226562, 251.1649932861328],
                [0, 0, 1],
            ],  # Color camera intrinsics
        }
        self.intrinsics = {k: np.array(v) for k, v in self.intrinsics.items()}

        self.z_near = 0.0
        self.z_far = 1.2

        self.lang_goal = instruction
        # Optional: Initialize tokenizer if needed
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def _resize_if_needed(self, image, size):
        """Resize the image if it does not match the desired size."""
        if image.size[0] != size[0] or image.size[1] != size[1]:
            image = image.resize(size)
        return image

    def _load_agent(self):
        """Load the agent configuration and weights."""
        cfg_path = os.path.join(self.cfg['agent']['seed_path'], 'config.yaml')
        cfg = OmegaConf.load(cfg_path)
        print('Agent configuration:', cfg)
        self.agent_cfg = cfg
        self.agent = any_bimanual.launch_utils.create_agent(cfg)

        self.agent.build(training=False, device=self.device)

        # Load pre-trained weights
        weights_path = os.path.join(self.cfg['agent']['seed_path'], 'weights',
                                    str(self.cfg['agent']['weight']))
        self.agent.load_weights(weights_path)

        print("Loaded weights from:", weights_path)

    def get_obs(self, img, depth, right_gripper_info, left_gripper_info, step):
        """
        Prepare the observation dictionary for the agent.

        Args:
            img (PIL.Image or np.ndarray): RGB image.
            depth (PIL.Image): Depth image.
            right_gripper_info (float): Right gripper state.
            left_gripper_info (float): Left gripper state.
            step (int): Current step in the episode.

        Returns:
            dict: Observation dictionary.
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        obs = {}
        # Pre-calibrated camera info
        front_camera_intrinsics = self.intrinsics['front_rgb']
        front_camera_extrinsics = self.extrinsics['front_rgb']
        obs['front_camera_intrinsics'] = torch.tensor([front_camera_intrinsics], device=self.device).unsqueeze(0)
        obs['front_camera_extrinsics'] = torch.tensor([front_camera_extrinsics], device=self.device).unsqueeze(0)

        # Real-time point cloud computation
        img = np.array(img)
        front_rgb = torch.tensor([img], device=self.device)   # Shape: [1, H, W, 3]
        front_rgb = front_rgb.permute(0, 3, 1, 2).unsqueeze(0)  # Shape: [1, 3, H, W]
        obs['front_rgb'] = front_rgb 

        front_depth = image_to_float_array(depth, DEPTH_SCALE)
        obs['front_depth'] = torch.tensor([front_depth], device=self.device)  # Shape: [1, H, W]

        near = self.z_near
        far = self.z_far
        front_depth_m = near + front_depth * (far - near)
        front_depth_m[front_depth_m > (far - 0.01)] = 0  # Thresholding invalid depth values

        front_point_cloud = pointcloud_from_depth_and_camera_params(
            front_depth_m,
            front_camera_extrinsics,
            front_camera_intrinsics
        )
        front_point_cloud = torch.tensor([front_point_cloud], device=self.device)
        front_point_cloud = front_point_cloud.permute(0, 3, 1, 2).unsqueeze(0)  # Shape: [1, 3, H, W]
        obs['front_point_cloud'] = front_point_cloud

        # Collision avoidance flag
        obs['ignore_collisions'] = torch.tensor([[[1.0]]], device=self.device)

        # Tokenize the language instruction
        inputs = tokenize(
            self.lang_goal,
            context_length=77  # Adjust context_length as needed
        )

        # Language inputs
        obs['lang_goal'] = self.lang_goal
        obs['lang_goal_tokens'] = inputs.to(self.device)

        # Gripper state
        right_finger_positions = right_gripper_info
        left_finger_positions = left_gripper_info
        threshold = 90
        right_gripper_open = 1.0 if right_finger_positions[0] < threshold else 0.0
        left_gripper_open = 1.0 if left_finger_positions[0] < threshold else 0.0
        right_finger_positions = right_finger_positions / 255.0
        left_finger_positions = left_finger_positions / 255.0

        # Normalize time step
        time_normalized = (1.0 - (step / float(self.cfg['agent']['episode_length'] - 1))) * 2.0 - 1.0
        
        right_low_dim_state = torch.tensor([[[right_gripper_open,
                                              right_finger_positions[0],
                                              right_finger_positions[1],
                                              time_normalized]]], device=self.device)
        
        left_low_dim_state = torch.tensor([[[left_gripper_open,
                                             left_finger_positions[0],
                                             left_finger_positions[1],
                                             time_normalized]]], device=self.device)
        
        obs['right_low_dim_state'] = right_low_dim_state
        obs['left_low_dim_state'] = left_low_dim_state
        return obs
    
    def adjust_gripper_z(self, action, displacement):
        """
        Adjust the gripper position along the Z-axis based on the given displacement.

        Args:
            action (list or np.ndarray): Original action containing position and orientation.
            displacement (float): Displacement along the Z-axis.

        Returns:
            np.ndarray: Adjusted action.
        """
        position = np.array(action[:3])  # x, y, z
        quaternion = np.array(action[3:7])  # rx, ry, rz, w
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()
        z_displacement = rotation_matrix[:, 2] * displacement 
        adjusted_position = position + z_displacement

        adjusted_action = np.concatenate([adjusted_position, quaternion, action[7:]])
        return adjusted_action[:8]

    def act(self, obs, step):
        """
        Generate actions based on the current observation.

        Args:
            obs (dict): Observation dictionary.
            step (int): Current step in the episode.

        Returns:
            tuple: Adjusted actions for right and left arms.
        """
        self.act_result = self.agent.act(step, obs, deterministic=True)

        action = self.act_result.action
        print(f"Raw action: {action}")

        right_act_res = action[:9]
        left_act_res = action[9:]
        right_act = self.adjust_gripper_z(right_act_res, -0.16)
        left_act = self.adjust_gripper_z(left_act_res, -0.16)
        # return right_act, left_act
        return right_act_res, left_act_res


# Mapping from task names to instructions
task_to_instruction = {
    'lift': 'lift the box.',    
    'pnp': 'put the green cube in the green box and put the red cube in the orange box.',
    'press': 'press dish soap into the bowl.',
    'handover': 'handover the bowl to the other hand.',
    'pick_in_one': 'place the two cubes in the bowl.',
    'pick_in_two': 'place the two cubes in boxes of the corresponding color.',
    'clothes': 'Fold the long-sleeve plaid shirt in half.',
    'pingpang': 'Perform a high-toss serve in table tennis.',
    'toothbrush': 'Pick up the toothbrush from one cup, flip it, and place it into another cup.',
    'pour': 'Pour the ping pong ball from one cup into another.',
    'robot': "Type 'robot' on the keyboard.",
}


def main():
    initialize(config_path="conf/real")
    config_name = "anybimanual_agent"
    cfg = compose(config_name=config_name)

    task_name = 'toothbrush'    # Specify the task name

    instruction = task_to_instruction[task_name]
    print(f"Task name: {task_name}, instruction: {instruction}")

    agent = AnyBimanualAgentInterface(cfg, instruction=instruction)
    agent._load_agent()
    episode_list = [7]  # List of episode indices to process

    right_total_loss = []
    left_total_loss = []
    for episode_idx in episode_list:
        
        # Place the path to your test data here
        episode_path = os.path.join('path_to_your_data', f'{task_name}_keyframe', 'all_variations', 'episodes', f'episode{episode_idx}')
        print(f"Episode path: {episode_path}")

        LOW_DIM_PICKLE = 'low_dim_obs.pkl'
        with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
            demo = pickle.load(f)

        episode_keypoints = [1, 2, 3]  # Key points in the episode to process

        test_num = len(episode_keypoints)

        step = 0

        for keypoint_idx in range(test_num):

            print('-----------------------------------')

            prev_step_idx = episode_keypoints[keypoint_idx - 1] if keypoint_idx != 0 else 0
            next_step_idx = episode_keypoints[keypoint_idx]
            print(f"Previous step index: {prev_step_idx}, next step index: {next_step_idx}, current step: {step}")

            image_path = os.path.join(episode_path, 'front_rgb', f'{prev_step_idx}.png')
            img = Image.open(image_path)
            
            # If the depth is preprocessed
            depth_path = image_path.replace('front_rgb', 'front_depth')
            depth = Image.open(depth_path)

            right_gripper_info = demo[prev_step_idx].right.gripper_joint_positions
            left_gripper_info = demo[prev_step_idx].left.gripper_joint_positions
            print(f"Right gripper info: {right_gripper_info}, left gripper info: {left_gripper_info}")
            # Prepare observation
            obs = agent.get_obs(img, depth, right_gripper_info, left_gripper_info, step)

            # Inference
            right_act_res, left_act_res = agent.act(obs, step)

            step += 1

            print(f"Right arm action: {right_act_res}, left arm action: {left_act_res}")

            # Get ground-truth actions
            right_gt_pose = demo[next_step_idx].right.gripper_pose
            left_gt_pose = demo[next_step_idx].left.gripper_pose
            right_gt_gripper = demo[next_step_idx].right.gripper_open
            left_gt_gripper = demo[next_step_idx].left.gripper_open

            right_gt_action = right_gt_pose + [right_gt_gripper, 1.0]
            left_gt_action = left_gt_pose + [left_gt_gripper, 1.0]
            print(f"Ground-truth right action: {right_gt_action}, left action: {left_gt_action}")
        
            right_total_loss.append(np.abs(np.array(right_act_res) - np.array(right_gt_action)))  # [8]
            left_total_loss.append(np.abs(np.array(left_act_res) - np.array(left_gt_action)))

    right_total_loss_mean = np.array(right_total_loss).mean(axis=0)
    left_total_loss_mean = np.array(left_total_loss).mean(axis=0)
    print(f"Average loss - Right arm: {right_total_loss_mean}, Left arm: {left_total_loss_mean}") 

    
if __name__ == "__main__":
    main()
