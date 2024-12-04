from abc import ABC, abstractmethod
from typing import Any, List
import inspect


class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class ScalarSummary(Summary):
    pass


class HistogramSummary(Summary):
    pass


class ImageSummary(Summary):
    pass


class TextSummary(Summary):
    pass


class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class ActResult(object):

    def __init__(self, action: Any,
                 observation_elements: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.observation_elements = observation_elements or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}


class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass


class BimanualAgent(Agent):
    """
    
    """

    def __init__(self, right_agent: Agent, left_agent: Agent):
        self.right_agent = right_agent
        self.left_agent = left_agent
        self._summaries = {}

    def build(self, training: bool, device=None) -> None:
        self.right_agent.build(training, device)
        self.left_agent.build(training, device)
    def update(self, step: int, replay_sample: dict) -> dict:
        right_observation = {}
        left_observation = {}

        for k, v in replay_sample.items(): 
            if "rgb" in k or "point_cloud" in k or "camera" in k:
                right_observation[k] = v
                left_observation[k] = v
            elif "right_" in k :
                right_observation[k[6:]] = v
            elif "left_" in k:
                left_observation[k[5:]] = v
            else:
                right_observation[k] = v
                left_observation[k] = v

        action = replay_sample["action"]
        right_action, left_action = action.chunk(2, dim=2)
        right_observation["action"] = right_action
        left_observation["action"] = left_action
        # qattention_peract_bc_agent
        right_update_dict = self.right_agent.update(step, right_observation)
        left_update_dict =  self.left_agent.update(step, left_observation)

        # relation between left and right
        total_losses = right_update_dict["total_losses"] + left_update_dict["total_losses"]
        self._summaries.update({"total_losses": total_losses})
        return self._summaries
    

    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:

        observation_elements = {}
        info = {}

        right_observation = {}
        left_observation = {}

        for k, v in observation.items(): 
            if "rgb" in k or "point_cloud" in k or "camera" in k:
                right_observation[k] = v
                left_observation[k] = v
            elif "right_" in k :
                right_observation[k[6:]] = v
            elif "left_" in k:
                left_observation[k[5:]] = v
            else:
                right_observation[k] = v
                left_observation[k] = v

        right_act_result = self.right_agent.act(step, right_observation, deterministic)
        left_act_result = self.left_agent.act(step, left_observation,deterministic)
        action = (*right_act_result.action, *left_act_result.action)

        observation_elements.update(right_act_result.observation_elements)
        observation_elements.update(left_act_result.observation_elements)

        info.update(right_act_result.info)
        info.update(left_act_result.info)

        return ActResult(action, observation_elements=observation_elements, info=info)

    def reset(self) -> None:
        self.right_agent.reset()
        self.left_agent.reset()

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for k, v in self._summaries.items():
            summaries.append(ScalarSummary(f"{k}", v))

        right_summaries = self.right_agent.update_summaries() 
        left_summaries =  self.left_agent.update_summaries()
        
        for summary in right_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_right/{summary.name}"

        for summary in left_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_left/{summary.name}"

        return right_summaries + left_summaries + summaries


    def act_summaries(self) -> List[Summary]:
        right_summaries = self.right_agent.act_summaries() 
        left_summaries =  self.left_agent.act_summaries()

        for summary in right_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_right/{summary.name}"

        for summary in left_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_left/{summary.name}"

        return right_summaries + left_summaries

    def load_weights(self, savedir: str) -> None:
        self.right_agent.load_weights(savedir)
        self.left_agent.load_weights(savedir)

    def save_weights(self, savedir: str) -> None:
        self.right_agent.save_weights(savedir)
        self.left_agent.save_weights(savedir)


class LeaderFollowerAgent(Agent):

    def __init__(self, leader_agent: Agent, follower_agent: Agent):
        self.leader_agent = leader_agent
        self.follower_agent = follower_agent
        self._summaries = {}

    def build(self, training: bool, device=None) -> None:
        self.leader_agent.build(training, device)
        self.follower_agent.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:

        leader_observation = {}
        follower_observation = {}

        for k, v in replay_sample.items(): 
            if "rgb" in k or "point_cloud" in k or "camera" in k:
                leader_observation[k] = v
                follower_observation[k] = v
            elif "right_" in k :
                leader_observation[k[6:]] = v
            elif "left_" in k:
                follower_observation[k[5:]] = v
            else:
                leader_observation[k] = v
                follower_observation[k] = v

        action = replay_sample["action"]
        right_action, left_action = action.chunk(2, dim=2)
        leader_observation["action"] = right_action
        follower_observation["action"] = left_action
        # leader -> right
        leader_update_dict = self.leader_agent.update(step, leader_observation)
        import torch
        follower_observation['low_dim_state'] = torch.cat([follower_observation['low_dim_state'],
                                                           replay_sample["right_trans_action_indicies"], 
                                                           replay_sample["right_rot_grip_action_indicies"], 
                                                           replay_sample["right_ignore_collisions"]], dim=-1)

        follower_update_dict = self.follower_agent.update(step, follower_observation)

        total_losses = leader_update_dict["total_losses"] + follower_update_dict["total_losses"]
        self._summaries.update({"total_losses": total_losses})
        return self._summaries
 
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:

        observation_elements = {}
        info = {}

        leader_observation = {}
        follower_observation = {}
        for k,v in observation.items():
            if "right_" in k and not "rgb" in k and not "point_cloud" in k and not "camera" in k:
                leader_observation[k[6:]] = v
            elif "left_" in k and not "rgb" in k and not "point_cloud" in k and not "camera" in k:
                follower_observation[k[5:]] = v
            else:
                leader_observation[k] = v
                follower_observation[k] = v
        # print(inspect.getfile(self.leader_agent.act))
        right_act_result = self.leader_agent.act(step, leader_observation, deterministic)

        right_observation_elements = right_act_result.observation_elements

        import torch

        device = follower_observation['low_dim_state'].device
        if "trans_action_indicies" in right_observation_elements:
            right_trans_action_indicies = torch.from_numpy(right_observation_elements["trans_action_indicies"]).unsqueeze(0).unsqueeze(0).to(device)
            right_rot_grip_action_indicies = torch.from_numpy(right_observation_elements["rot_grip_action_indicies"]).unsqueeze(0).unsqueeze(0).to(device)
            right_ignore_collisions = torch.from_numpy(right_act_result.action[-1:]).unsqueeze(0).unsqueeze(0).to(device)
        else:
            right_trans_action_indicies = torch.empty((1, 1, 3)).to(device)
            right_rot_grip_action_indicies = torch.empty((1, 1, 4)).to(device)
            right_ignore_collisions = torch.empty((1, 1, 1)).to(device)
 

        follower_observation['low_dim_state'] = torch.cat([follower_observation['low_dim_state'],
                                                           right_trans_action_indicies,
                                                           right_rot_grip_action_indicies,
                                                           right_ignore_collisions], dim=-1)  
        
        left_act_result = self.follower_agent.act(step, follower_observation, deterministic)

        action = (*right_act_result.action, *left_act_result.action)
        
        observation_elements.update(right_act_result.observation_elements)
        observation_elements.update(left_act_result.observation_elements)

        info.update(right_act_result.info)
        info.update(left_act_result.info)

        return ActResult(action, observation_elements=observation_elements, info=info)
    

    def reset(self) -> None:
        self.leader_agent.reset()
        self.follower_agent.reset()

    def update_summaries(self) -> List[Summary]:

        summaries = []
        for k, v in self._summaries.items():
            summaries.append(ScalarSummary(f"{k}", v))

        leader_summaries = self.leader_agent.update_summaries() 
        follower_summaries =  self.follower_agent.update_summaries()

        for summary in leader_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_leader/{summary.name}"
        for summary in follower_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_follower/{summary.name}"

        return leader_summaries + follower_summaries + summaries
        

    def act_summaries(self) -> List[Summary]:
        leader_summaries = self.leader_agent.act_summaries() 
        follower_summaries =  self.follower_agent.act_summaries()

        for summary in leader_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_leader/{summary.name}"
        for summary in follower_summaries:
            if not isinstance(summary, ImageSummary):
                summary.name = f"agent_follower/{summary.name}"

        return leader_summaries + follower_summaries

    def load_weights(self, savedir: str) -> None:
        self.leader_agent.load_weights(savedir)
        self.follower_agent.load_weights(savedir)

    def save_weights(self, savedir: str) -> None:
        self.leader_agent.save_weights(savedir)
        self.follower_agent.save_weights(savedir)
