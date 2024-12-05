import os
from typing import List
import torch
import numpy as np

from omegaconf import DictConfig

from yarr.agents.agent import Agent
from yarr.agents.agent import ActResult
from yarr.agents.agent import Summary
from yarr.agents.agent import ScalarSummary
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
from helpers.preprocess_agent import PreprocessAgent
from agents.rvt.rvt.models.skill_manager import SkillManager
from agents.rvt.rvt.models.visual_aligner import VisualAligner

from agents.rvt.rvt.mvt.mvt import MVT
from agents.rvt.rvt.models import rvt_agent
from agents.rvt.rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)


import agents.rvt.rvt.config as exp_cfg_mod
import agents.rvt.rvt.models.rvt_agent as rvt_agent
import agents.rvt.rvt.mvt.config as mvt_cfg_mod
import os

def create_agent(cfg: DictConfig):

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    exp_cfg.bs = cfg.replay.batch_size
    exp_cfg.tasks = ','.join(cfg.rlbench.tasks)
    
    exp_cfg.freeze()

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    mvt_cfg.proprio_dim = cfg.method.low_dim_size
    mvt_cfg.freeze()

    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    pkl_path = os.path.join(current_dir, "../../lang_token.pkl")
    pkl_path = os.path.abspath(pkl_path)
    with open(pkl_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    flattened_embeddings = []
    for key in embeddings_dict.keys():
        embedding = torch.tensor(embeddings_dict[key]) 
        flattened_embedding = embedding.view(-1) 
        flattened_embeddings.append(flattened_embedding)
    embeddings_matrix = torch.stack(flattened_embeddings)  
    skill_manager = SkillManager(num_classes=18,embedding_matrix=embeddings_matrix)
    visual_aligner = VisualAligner()
    agent = RVTAgentWrapper(cfg.framework.checkpoint_name_prefix, cfg.rlbench, mvt_cfg, exp_cfg, skill_manager, visual_aligner)


    preprocess_agent = PreprocessAgent(pose_agent=agent)
    return preprocess_agent



class RVTAgentWrapper(Agent):

    def __init__(self, checkpoint_name_prefix, rlbench_cfg, mvt_cfg, exp_cfg, skill_manager, visual_aligner):
        self._checkpoint_filename = f"{checkpoint_name_prefix}.pt"
        self.rvt_agent = None
        self.rlbench_cfg = rlbench_cfg
        self.mvt_cfg = mvt_cfg
        self.exp_cfg = exp_cfg
        self._summaries = {}
        self.skill_manager = skill_manager
        self.visual_aligner = visual_aligner
        
    def build(self, training: bool, device=None) -> None:

        import torch
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        self._device = device
        if isinstance(device, int):
            device = f"cuda:{device}"

        rvt = MVT(
            renderer_device=device,
            **self.mvt_cfg,
        )
        rvt = rvt.to(device)

        if training:
            rvt = DDP(rvt, device_ids=[device])

        self.rvt_agent = rvt_agent.RVTAgent(
            network=rvt,
            #image_resolution=self.rlbench_cfg.camera_resolution,
            skill_manager=self.skill_manager,
            visual_aligner=self.visual_aligner,
            stage_two=False,
            add_lang=self.mvt_cfg.add_lang,
            scene_bounds=self.rlbench_cfg.scene_bounds,
            cameras=self.rlbench_cfg.cameras,
            log_dir="/tmp/eval_run",
            **self.exp_cfg.peract,
            **self.exp_cfg.rvt,

        )

        self.rvt_agent.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        for k, v in replay_sample.items():
            replay_sample[k] = v.unsqueeze(1)
        # RVT is based on the PerAct's Colab version.
        replay_sample["lang_goal_embs"] = replay_sample["lang_token_embs"]
        replay_sample["tasks"] = self.exp_cfg.tasks.split(',')
        
        update_dict = self.rvt_agent.update(step, replay_sample)


        for key, val in self.rvt_agent.loss_log.items():
            self._summaries[key] = np.mean(np.array(val))
        device = self._device
        rank = device
        if step % 10 == 0 and rank == 0:
            wandb.log({
                'train/grip_loss': update_dict["grip_loss"],
                'train/trans_loss': update_dict["trans_loss"],
                'train/rot_loss': (update_dict["rot_loss_x"]+update_dict["rot_loss_y"]+update_dict["rot_loss_z"]),
                'train/collision_loss': update_dict["collision_loss"],
                'train/total_loss': update_dict["total_loss"],
            }, step=step)
        self._wandb_summaries = {
                'losses/grip_loss': update_dict["grip_loss"],
                'losses/trans_loss': update_dict["trans_loss"],
                'losses/rot_loss': (update_dict["rot_loss_x"]+update_dict["rot_loss_y"]+update_dict["rot_loss_z"]),
                'losses/collision_loss': update_dict["collision_loss"],
                'losses/total_loss': update_dict["total_loss"],
        }
        return {
            "total_losses": update_dict["total_loss"],
        }

        return result

    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        return self.rvt_agent.act(step, observation, deterministic)

    def reset(self) -> None:
        self.rvt_agent.reset()

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for k, v in self._summaries.items():
            summaries.append(ScalarSummary(f"RVT/{k}", v))
        return summaries

    def update_wandb_summaries(self):
        summaries = dict()

        for k, v in self._wandb_summaries.items():
            summaries[k] = v
        return summaries

    def act_summaries(self) -> List[Summary]:
        return []
    
    def load_weights(self, savedir: str) -> None:
        """
        copied from RVT
        """
        device = torch.device("cuda:0")
        weight_file = os.path.join(savedir, self._checkpoint_filename)
        state_dict = torch.load(weight_file, map_location=device)

        skill = self.rvt_agent.skill_manager
        visual_aligner = self.rvt_agent.visual_aligner
        model = self.rvt_agent._network
        optimizer = self.rvt_agent._optimizer
        lr_sched = self.rvt_agent._lr_sched

        if isinstance(model, DDP):
            model = model.module 
        model.load_state_dict(state_dict["model_state"])
        optimizer.load_state_dict(state_dict["optimizer_state"])
        lr_sched.load_state_dict(state_dict["lr_sched_state"])
    
        return self.rvt_agent.load_clip()
        
   
    def save_weights(self, savedir: str) -> None:

        os.makedirs(savedir, exist_ok=True)
        weight_file = os.path.join(savedir, self._checkpoint_filename)
        skill = self.rvt_agent.skill_manager
        visual_aligner = self.rvt_agent.visual_aligner
        model = self.rvt_agent._network
        optimizer = self.rvt_agent._optimizer
        lr_sched = self.rvt_agent._lr_sched

        if isinstance(model, DDP):
            model = model.module
        
        skill_state = skill.state_dict()
        visual_aligner_state = visual_aligner.state_dict()
        model_state = model.state_dict()

        torch.save(
            {
                "skill_state": skill_state,
                "visual_aligner_state": visual_aligner_state,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "lr_sched_state": lr_sched.state_dict(),
            },
            weight_file,
        )

