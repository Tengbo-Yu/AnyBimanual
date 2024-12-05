import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class VisualAligner(nn.Module):
    def __init__(self):
        super(VisualAligner, self).__init__()
        
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()

        self.encoder_pcd = self._make_encoder()
        self.decoder_pcd = self._make_decoder()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _make_encoder(self):
        return nn.Sequential(
            ResidualBlock(3, 64), 
            nn.MaxPool2d(2),
            ResidualBlock(64, 128), 
            nn.MaxPool2d(2)
        )

    def _make_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64), 
            nn.ConvTranspose2d(64, 2, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, rgb_list, pcd_list):
        rgb_right, rgb_left, pcd_right, pcd_left = [], [], [], []
        for rgb in rgb_list:
            rgb = rgb.to(self.device) 
            mask = self.decoder(self.encoder(rgb)) 
            mrr, mrl = self._apply_mask(rgb, mask)
            rgb_right.append(mrr)
            rgb_left.append(mrl)

        for pcd in pcd_list:
            pcd = pcd.to(self.device)
            mask = self.decoder_pcd(self.encoder_pcd(pcd)) 
            mpr, mpl = self._apply_mask(pcd, mask)
            pcd_right.append(mpr)
            pcd_left.append(mpl)

        return rgb_right, rgb_left, pcd_right, pcd_left

    def _apply_mask(self, input_tensor, mask):
        mask_right = mask[:, 0:1, :, :].expand(-1, 3, -1, -1)
        mask_left = mask[:, 1:2, :, :].expand(-1, 3, -1, -1)
        masked_right = input_tensor * mask_right
        masked_left = input_tensor * mask_left
        return masked_right, masked_left