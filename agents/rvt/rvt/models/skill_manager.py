import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from agents.rvt.rvt.models.trajectory_gpt2 import GPT2Model

class SkillManager(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_matrix,
            voxel_dim=64,
            lang_dim=64,
            hidden_size=128,
            output_dim=18,
            max_voxels=200,
            max_lang_tokens=77
    ):
        super(SkillManager, self).__init__()

        self.num_class = num_classes
        self.output_dim = output_dim
        self.max_voxels = max_voxels
        self.max_lang_tokens = max_lang_tokens
        self.hidden_size = hidden_size
        self.voxel_dim = voxel_dim
        self.lang_dim = lang_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2), 
            nn.AdaptiveAvgPool2d((8,8)) 
        )

        cnn_output_dim = 64 * 8 * 8 

        total_embedding_dim = (max_voxels * voxel_dim) + (max_lang_tokens * lang_dim) 
        self.embedding_projector = nn.Linear(cnn_output_dim, total_embedding_dim)

        config = transformers.GPT2Config(
            vocab_size=1, 
            n_embd=hidden_size,
            n_head=4,
            n_ctx=1077,
        )

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.transformer = GPT2Model(config)
        self.predict_logits = nn.Linear(hidden_size, output_dim)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.embeddings_matrix = embedding_matrix.to(self.device)

        if self.voxel_dim != self.hidden_size or self.lang_dim != self.hidden_size:
            self.projection_layer = nn.Linear(self.voxel_dim, self.hidden_size).to(self.device)
        else:
            self.projection_layer = None

    def forward(self, rgb_list, pcd_list):
        combined = torch.cat((rgb_list[0].to(self.device), pcd_list[0].to(self.device)), dim=1)
        for i in range(1, len(rgb_list)):
            combined = torch.cat((combined, rgb_list[i].to(self.device), pcd_list[i].to(self.device)), dim=1)

        features = self.conv(combined) 
        flat = features.view(features.size(0), -1) 

        emb_output = self.embedding_projector(flat) 
        voxel_size = self.max_voxels * self.voxel_dim  
        lang_size = self.max_lang_tokens * self.lang_dim
        voxel_emb = emb_output[:, :voxel_size].view(-1, self.max_voxels, self.voxel_dim)
        lang_emb = emb_output[:, voxel_size:voxel_size+lang_size].view(-1, self.max_lang_tokens, self.lang_dim)

        batch_size = voxel_emb.shape[0]

        voxel_emb = voxel_emb.permute(0, 2, 1)  
        voxel_emb = F.avg_pool1d(voxel_emb, kernel_size=10, stride=10)
        voxel_emb = voxel_emb.permute(0, 2, 1)  

        inputs = torch.cat([lang_emb, voxel_emb], dim=1)

        if self.projection_layer is not None:
            inputs = self.projection_layer(inputs)

        stacked_inputs = self.embed_ln(inputs)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=None,
        )

        hidden_state = transformer_outputs.last_hidden_state
        aggregated_hidden = hidden_state.mean(dim=1)
        logits = self.predict_logits(aggregated_hidden)
        probs = F.softmax(logits, dim=1)
        skill = torch.matmul(probs, self.embeddings_matrix.to(probs.device))
        skill = skill.view(-1,77,512)

        return skill