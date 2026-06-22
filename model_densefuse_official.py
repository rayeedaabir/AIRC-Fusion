"""
model_densefuse_official.py  (NEW - FAITHFUL DenseFuse [47], official architecture)
-----------------------------------------------------------------------------------
Replaces the old "densefuse" baseline, whose model.py was a different
(concatenate-input) network, NOT DenseFuse. This embeds the authors' official
PyTorch architecture verbatim (github.com/hli1221/densefuse-pytorch, net.py):
encoder (conv 1->16 + dense block ->64) -> fusion (addition or L1-norm) -> decoder
(64->64->32->16->1). Runs on CPU. Model key: "densefuse_official".

ONE-TIME SETUP (you do this):
  1. Download the pretrained weight file from the densefuse-pytorch repo's
     `models/` folder (a *.model / *.pth, grayscale, input_nc=1).
  2. Put it at  models/densefuse.model  in this project (or set DENSEFUSE_WEIGHTS).
Disclose in the paper: "DenseFuse [47], authors' official PyTorch architecture and
pretrained weights; fusion strategy = L1-norm."
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

DENSEFUSE_WEIGHTS = os.environ.get("DENSEFUSE_WEIGHTS",
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "densefuse.model"))


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride, is_last=False):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(k / 2)))
        self.conv2d = nn.Conv2d(in_c, out_c, k, stride)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))
        return out if self.is_last else F.relu(out, inplace=True)


class DenseConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, stride):
        super().__init__()
        self.dense_conv = ConvLayer(in_c, out_c, k, stride)

    def forward(self, x):
        return torch.cat([x, self.dense_conv(x)], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_c, k, stride):
        super().__init__()
        d = 16
        self.denseblock = nn.Sequential(
            DenseConv2d(in_c, d, k, stride),
            DenseConv2d(in_c + d, d, k, stride),
            DenseConv2d(in_c + d * 2, d, k, stride))

    def forward(self, x):
        return self.denseblock(x)


class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super().__init__()
        nb = [16, 64, 32, 16]; k = 3; s = 1
        self.conv1 = ConvLayer(input_nc, nb[0], k, s)
        self.DB1 = DenseBlock(nb[0], k, s)
        self.conv2 = ConvLayer(nb[1], nb[1], k, s)
        self.conv3 = ConvLayer(nb[1], nb[2], k, s)
        self.conv4 = ConvLayer(nb[2], nb[3], k, s)
        self.conv5 = ConvLayer(nb[3], output_nc, k, s)

    def encoder(self, x):
        return [self.DB1(self.conv1(x))]

    def decoder(self, f):
        return [self.conv5(self.conv4(self.conv3(self.conv2(f[0]))))]


def _l1_fuse(en1, en2):
    """DenseFuse L1-norm (soft) fusion of feature maps."""
    a1 = torch.sum(torch.abs(en1), dim=1, keepdim=True)
    a2 = torch.sum(torch.abs(en2), dim=1, keepdim=True)
    # box-average the activity maps (window-based, as in the paper)
    kernel = torch.ones(1, 1, 3, 3, device=en1.device) / 9.0
    a1 = F.conv2d(F.pad(a1, (1, 1, 1, 1), mode="reflect"), kernel)
    a2 = F.conv2d(F.pad(a2, (1, 1, 1, 1), mode="reflect"), kernel)
    w1 = a1 / (a1 + a2 + 1e-8)
    return w1 * en1 + (1.0 - w1) * en2


class DenseFuseOfficial:
    def __init__(self, weights_path=DENSEFUSE_WEIGHTS, strategy="l1", device="cpu"):
        self.device = torch.device(device)
        self.strategy = strategy
        self.model = DenseFuse_net(1, 1)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [weights_path,
                      os.path.join(here, "models", "densefuse.model"),
                      os.path.join(here, "models", "densefuse_gray.model"),
                      os.path.join(here, "densefuse_gray.model"),
                      os.path.join(here, "densefuse.model")]
        weights_path = next((c for c in candidates if c and os.path.exists(c)), None)
        if weights_path is None:
            raise FileNotFoundError(
                "DenseFuse weights not found. Use the GRAYSCALE file (densefuse_gray.model) "
                "from github.com/hli1221/densefuse-pytorch and put it at models/densefuse.model "
                "(or anywhere listed in model_densefuse_official.py), or set DENSEFUSE_WEIGHTS.")
        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"DenseFuse (official) loaded from {weights_path}")

    def fuse(self, visible_uint8_bgr, infrared_uint8_bgr):
        vis_ycrcb = cv2.cvtColor(visible_uint8_bgr, cv2.COLOR_BGR2YCrCb)
        ir_gray = cv2.cvtColor(infrared_uint8_bgr, cv2.COLOR_BGR2GRAY)
        vy = vis_ycrcb[:, :, 0].astype(np.float32) / 255.0
        iy = ir_gray.astype(np.float32) / 255.0
        t_v = torch.from_numpy(vy)[None, None].to(self.device)
        t_i = torch.from_numpy(iy)[None, None].to(self.device)
        with torch.no_grad():
            en_v = self.model.encoder(t_v)
            en_i = self.model.encoder(t_i)
            if self.strategy == "add":
                fused = [(en_v[0] + en_i[0]) / 2.0]
            else:
                fused = [_l1_fuse(en_v[0], en_i[0])]
            out_y = self.model.decoder(fused)[0]
        y = np.clip(out_y.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        # carry colour from the visible image (IR is monochromatic)
        out = vis_ycrcb.copy()
        out[:, :, 0] = y
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
