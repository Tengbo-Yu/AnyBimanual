import torch
import torch.nn as nn
import transformers
from agents.peract_bimanual.trajectory_gpt2 import GPT2Model
import torch.nn.functional as F
class SkillManager(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_matrix=None,
            voxel_dim=128,
            lang_dim=128,
            hidden_size=128,
            output_dim=18,
            max_voxels=8000,
            max_lang_tokens=77,
            **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # GPT-2 configuration
        config = transformers.GPT2Config(
            vocab_size=1,  # not used
            n_embd=hidden_size,
            n_head=4, 
            n_ctx=1077,
        )

        self.max_voxels = max_voxels
        self.max_lang_tokens = max_lang_tokens
        self.embed_voxel = nn.Linear(voxel_dim, hidden_size)
        self.embed_lang = nn.Linear(lang_dim, hidden_size)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_logits = nn.Linear(hidden_size, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes
        if embedding_matrix is not None:
            self.embeddings_matrix = embedding_matrix.to(self.device)

    def forward(self, voxel_embedding, language_embedding):
        batch_size = voxel_embedding.shape[0]
        voxel_embeddings = self.embed_voxel(voxel_embedding)  # [b, 8000, hidden_size]
        language_embeddings = self.embed_lang(language_embedding)  # [b, 77, hidden_size]
        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, hidden_size, 8000]
        voxel_embeddings = F.avg_pool1d(voxel_embeddings, kernel_size=16, stride=16)  # [b, hidden_size, 1000]
        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, 1000, hidden_size]
        inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)  # [b, 8077, hidden_size]
        stacked_inputs = self.embed_ln(inputs)
        attention_mask = torch.ones(
            (batch_size, self.max_lang_tokens + self.max_voxels),
            device=voxel_embedding.device,
            dtype=torch.long  # Ensure correct dtype
        )
        assert torch.isfinite(attention_mask).all(), "attention_mask contains NaN or Inf"
        assert torch.all((attention_mask == 1)), "attention_mask contains values not equal to 1"
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=None,
        )

        hidden_state = transformer_outputs.last_hidden_state  # [b, 8077, hidden_size]
        aggregated_hidden = hidden_state.mean(dim=1)  # [b, hidden_size]
        logits = self.predict_logits(aggregated_hidden)  # [b, output_dim]
        probs = F.softmax(logits, dim=1)
        skill = torch.matmul(probs, self.embeddings_matrix.to(probs.device))
        skill = skill.view(-1,77,512)
        return skill