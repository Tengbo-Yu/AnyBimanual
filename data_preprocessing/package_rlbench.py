import random
import itertools
from typing import Tuple, Dict, List
import pickle
from pathlib import Path
import json

import blosc
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo

from utils.utils_with_rlbench import (
    RLBenchEnv,
    keypoint_discovery,
    obs_to_attn,
    obs_to_attn_right,
    obs_to_attn_left,
    transform,
)


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"
    seed: int = 2
    tasks: Tuple[str, ...] = ("stack_wine",)
    cameras: Tuple[str, ...] = ("over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front")
    image_size: str = "256,256"
    output: Path = Path(__file__).parent / "datasets"
    max_variations: int = 199
    offset: int = 0
    num_workers: int = 0
    store_intermediate_actions: int = 1


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...], Arm: str
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    frames.insert(0, 0)
    if Arm == 'right':
        right_cameras = ( "over_shoulder_right", "overhead", "wrist_right", "front")
        return [{cam: obs_to_attn_right(demo[f], cam) for cam in right_cameras} for f in frames]
    else:
        left_cameras = ("over_shoulder_left", "overhead", "wrist_left", "front")
        return [{cam: obs_to_attn_left(demo[f], cam) for cam in left_cameras} for f in frames]

def get_observation(task_str: str, variation: int,
                    episode: int, env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    key_frame = keypoint_discovery(demo)
    key_frame.insert(0, 0)

    # keyframe_state_ls = []
    right_keyframe_state_ls = []
    left_keyframe_state_ls = []
    right_keyframe_action_ls = []
    left_keyframe_action_ls = []
    right_intermediate_action_ls = []
    left_intermediate_action_ls = []

    for i in range(len(key_frame)):
        right_state, right_action = env.get_obs_action_right(demo._observations[key_frame[i]])
        left_state, left_action = env.get_obs_action_left(demo._observations[key_frame[i]])
        # state = transform(state)
        right_state = transform(right_state)
        left_state = transform(left_state)
        # keyframe_state_ls.append(state.unsqueeze(0))
        right_keyframe_state_ls.append(right_state.unsqueeze(0))
        left_keyframe_state_ls.append(left_state.unsqueeze(0))
        right_keyframe_action_ls.append(right_action.unsqueeze(0))
        left_keyframe_action_ls.append(left_action.unsqueeze(0))
        if store_intermediate_actions and i < len(key_frame) - 1:
            right_intermediate_actions = []
            left_intermediate_actions = []
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, right_action= env.get_obs_action_right(demo._observations[j])
                _, left_action= env.get_obs_action_left(demo._observations[j])
                right_intermediate_actions.append(right_action.unsqueeze(0))
                left_intermediate_actions.append(left_action.unsqueeze(0))
            right_intermediate_action_ls.append(torch.cat(right_intermediate_actions))
            left_intermediate_action_ls.append(torch.cat(left_intermediate_actions))

    return demo, right_keyframe_state_ls, left_keyframe_state_ls, right_keyframe_action_ls, left_keyframe_action_ls, right_intermediate_action_ls, left_intermediate_action_ls


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[256,256],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        tasks = args.tasks
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        (demo,
         right_keyframe_state_ls,
         left_keyframe_state_ls,
         right_keyframe_action_ls,
         left_keyframe_action_ls,
         right_intermediate_action_ls,
         left_intermediate_action_ls) = get_observation(
            task, variation, episode, self.env,
            bool(args.store_intermediate_actions)
        )

        right_state_ls = einops.rearrange(
            right_keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=4, # len(right_cameras)
            m=2,
        )

        left_state_ls = einops.rearrange(
            left_keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=4, # len(left_cameras)
            m=2,
        )

        right_frame_ids = list(range(len(right_state_ls) - 1))
        left_frame_ids = list(range(len(left_state_ls) - 1))
        # num_frames = len(frame_ids)
        right_attn_indices = get_attn_indices_from_demo(task, demo, args.cameras,'right')
        left_attn_indices = get_attn_indices_from_demo(task, demo, args.cameras,'left')


        # unimanual
        # state_dict: List = [[] for _ in range(6)]
        # print("Demo {}".format(episode))
        # state_dict[0].extend(frame_ids)
        # state_dict[1] = state_ls[:-1].numpy()
        # state_dict[2].extend(keyframe_action_ls[1:])
        # state_dict[3].extend(attn_indices)
        # state_dict[4].extend(keyframe_action_ls[:-1])  # gripper pos
        # state_dict[5].extend(intermediate_action_ls)   # traj from gripper pos to keyframe action
        # bimanual
        state_dict: List = [[] for _ in range(12)]
        print("Demo {}".format(episode))
        state_dict[0].extend(right_frame_ids)
        state_dict[1].extend(left_frame_ids)
        state_dict[2] = right_state_ls[:-1].numpy()
        state_dict[3] = left_state_ls[:-1].numpy()
        state_dict[4].extend(right_keyframe_action_ls[1:]) # right action
        state_dict[5].extend(left_keyframe_action_ls[1:]) # left action
        state_dict[6].extend(right_attn_indices)
        state_dict[7].extend(left_attn_indices)
        state_dict[8].extend(right_keyframe_action_ls[:-1])  # right gripper pos
        state_dict[9].extend(left_keyframe_action_ls[:-1])  # left gripper pos
        state_dict[10].extend(right_intermediate_action_ls)   # traj from gripper pos to keyframe action
        state_dict[11].extend(left_intermediate_action_ls)   # traj from gripper pos to keyframe action

        with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue