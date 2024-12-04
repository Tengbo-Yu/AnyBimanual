import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualAligner(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, mask_dim=128):
        super(VisualAligner, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        self.conv_res1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv_res2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.conv2_right = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, ins):
        ins = ins.transpose(1, 2)

        features = self.activation(self.conv1(ins))

        residual = features
        features = self.activation(self.conv_res1(features))
        features = self.conv_res2(features)
        features = features + residual 

        mask_right = self.activation(self.conv2_right(features))
        mask_left = self.activation(self.conv2_left(features))

        mask_right = mask_right.transpose(1, 2)
        mask_left = mask_left.transpose(1, 2)
        ins = ins.transpose(1, 2)

        masked_ins1 = ins * mask_right
        masked_ins2 = ins * mask_left

        return masked_ins1, masked_ins2