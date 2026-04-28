#!/usr/bin/env python3
"""
Split-task depth reconstruction with dual branch (building / ground).
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_IO_SHOW_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import math
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from datetime import datetime
import torch.nn.functional as F
import gc
import time
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
TS = 512


# =====================================================================
#  Core Modules
# =====================================================================

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU() if act else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MambaBlock2D(nn.Module):
    def __init__(self, ch, expansion=2, ks=7, dropout=0.1):
        super().__init__()
        h = ch * expansion
        self.norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.in_proj = nn.Conv2d(ch, h * 2, 1, bias=False)
        self.dw = nn.Conv2d(h, h, ks, padding=ks // 2, groups=h, bias=False)
        self.act = nn.SiLU()
        self.out_proj = nn.Conv2d(h, ch, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.kaiming_normal_(self.in_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='linear')
    def forward(self, x):
        r = x
        x = self.norm(x)
        x = self.in_proj(x)
        u, v = x.chunk(2, dim=1)
        u = self.act(u)
        v = self.dw(v)
        x = u * v
        x = self.out_proj(x)
        x = self.drop(x)
        return r + self.gamma * x


class MambaStage(nn.Module):
    def __init__(self, in_c, out_c, depth, down=True, dropout=0.1):
        super().__init__()
        self.down = ConvLayer(in_c, out_c, s=2 if down else 1)
        self.blocks = nn.Sequential(*[MambaBlock2D(out_c, dropout=dropout) for _ in range(depth)])
    def forward(self, x):
        return self.blocks(self.down(x))


class MambaEncoder(nn.Module):
    def __init__(self, in_c=6, dm=64, depths=(3, 4, 3), dropout=0.1):
        super().__init__()
        self.stem = ConvLayer(in_c, 64, k=4, s=2, p=1)
        self.s1 = MambaStage(64, 64, depths[0], True, dropout)
        self.s2 = MambaStage(64, 128, depths[1], True, dropout)
        self.s3 = MambaStage(128, 256, depths[2], True, dropout)
        self.p1 = ConvLayer(64, dm, k=1, p=0)
        self.p2 = ConvLayer(128, dm, k=1, p=0)
        self.p3 = ConvLayer(256, dm, k=1, p=0)
    def forward(self, x):
        x = self.stem(x)
        x = torch.clamp(x, -10.0, 10.0)
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return [self.p1(f1), self.p2(f2), self.p3(f3)]


class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(),
            nn.Conv2d(c // r, c, 1, bias=False))
        self.sp = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        ca = self.sig(self.fc(self.avg(x)) + self.fc(self.mx(x)))
        x = x * ca
        sa = self.sig(self.sp(torch.cat([x.mean(1, True), x.max(1)[0].unsqueeze(1)], 1)))
        return x * sa


class PositionalEncoding(nn.Module):
    def __init__(self, dm=64):
        super().__init__()
        self.dm = dm
        self.register_buffer('pe', torch.zeros(1, dm, 1, 1), persistent=False)
        self.sz = (1, 1)
    def forward(self, x):
        _, _, H, W = x.shape
        if self.sz != (H, W) or self.pe.device != x.device:
            px = torch.arange(W, dtype=x.dtype, device=x.device).repeat(H, 1)
            py = torch.arange(H, dtype=x.dtype, device=x.device).repeat(W, 1).t()
            pe = torch.zeros(1, self.dm, H, W, dtype=x.dtype, device=x.device)
            d = torch.exp(torch.arange(0, self.dm, 2, device=x.device,
                                       dtype=x.dtype) * (-math.log(10000.0) / self.dm))
            pe[0, ::2] = (px.unsqueeze(0) * d[:, None, None]).sin()
            pe[0, 1::2] = (py.unsqueeze(0) * d[:, None, None]).cos()
            self.pe = pe
            self.sz = (H, W)
        return x + self.pe


class LightCrossAttention(nn.Module):
    def __init__(self, dm=64, nh=8, bs=8, drop=0.1):
        super().__init__()
        assert dm % nh == 0
        self.nh, self.bs, self.hd = nh, bs, dm // nh
        self.sc = self.hd ** (-0.5)
        self.drop = nn.Dropout(drop)
        self.proj_q = nn.Conv2d(dm, dm, 1)
        self.proj_k = nn.Conv2d(dm, dm, 1)
        self.proj_v = nn.Conv2d(dm, dm, 1)
        self.proj_out = nn.Conv2d(dm, dm, 1)
        self.norm = nn.BatchNorm2d(dm)
    def forward(self, q, k, v):
        B, C, H, W = q.shape
        q_orig = q
        bh, bw = H // self.bs, W // self.bs
        if bh == 0 or bw == 0:
            return self.norm(self.proj_out(self.proj_q(q)) + q)
        nb = bh * bw
        n = self.bs * self.bs
        def blockify(x):
            return x.reshape(B, C, bh, self.bs, bw, self.bs).permute(
                0, 2, 4, 1, 3, 5).reshape(B * nb, C, self.bs, self.bs)
        qb = self.proj_q(blockify(q))
        kb = self.proj_k(blockify(k))
        vb = self.proj_v(blockify(v))
        Bb = qb.shape[0]
        qb = qb.reshape(Bb, self.nh, self.hd, n)
        kb = kb.reshape(Bb, self.nh, self.hd, n)
        vb = vb.reshape(Bb, self.nh, self.hd, n)
        attn = F.softmax(torch.einsum('bhdn,bhdm->bhnm', qb, kb) * self.sc, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, vb)
        out = out.contiguous().reshape(Bb, C, self.bs, self.bs)
        out = self.proj_out(out)
        out = out.reshape(B, bh, bw, C, self.bs, self.bs).permute(
            0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return self.norm(out + q_orig)


class LightTransformerBlock(nn.Module):
    def __init__(self, dm=64, nh=8, hd=128, drop=0.1):
        super().__init__()
        self.self_attn = LightCrossAttention(dm, nh, bs=8, drop=drop)
        self.ffn = nn.Sequential(
            nn.Conv2d(dm, hd, 1), nn.GELU(), nn.Dropout(drop),
            nn.Conv2d(hd, dm, 1))
        self.norm = nn.BatchNorm2d(dm)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x + self.drop(self.ffn(self.norm(x)))


class DynamicFusionGate(nn.Module):
    def __init__(self, c, h=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(c * 3, h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1))
    def forward(self, a, b, c):
        return torch.sigmoid(self.gate(torch.cat([a, b, c], 1)))


class EdgeRefinementHead(nn.Module):
    def __init__(self, in_ch=2, h=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, h, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, groups=h), nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, 3, padding=1))
    def forward(self, pred, edge):
        return pred + self.block(torch.cat([pred, edge], 1))


def image_edge_map(img):
    g = img.mean(1, True)
    gx = torch.zeros_like(g)
    gy = torch.zeros_like(g)
    gx[:, :, :, 1:] = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs()
    gy[:, :, 1:, :] = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs()
    return gx + gy


def sobel_edges(t):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=t.device, dtype=t.dtype).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      device=t.device, dtype=t.dtype).view(1, 1, 3, 3)
    return F.conv2d(t, sx, padding=1), F.conv2d(t, sy, padding=1)


def make_boundary_mask(bm_bin, valid_mask=None, k=5):
    if bm_bin.dim() == 3:
        bm_bin = bm_bin.unsqueeze(1)
    if valid_mask is None:
        valid_mask = torch.ones_like(bm_bin)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    pad = k // 2
    dil = F.max_pool2d(bm_bin, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - bm_bin, kernel_size=k, stride=1, padding=pad)
    bd = (dil - ero).clamp(0.0, 1.0)
    return (bd > 0.01).float() * valid_mask.float()


# =====================================================================
#  Decoder
# =====================================================================

class PixelShuffleDecoder(nn.Module):
    def __init__(self, dm=64, task='building'):
        super().__init__()
        self.task = task
        self.up1 = nn.Sequential(
            nn.Conv2d(dm, 128 * 4, 3, padding=1), nn.PixelShuffle(2),
            nn.BatchNorm2d(128), nn.GELU())
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1), nn.PixelShuffle(2),
            nn.BatchNorm2d(64), nn.GELU())
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, 3, padding=1), nn.PixelShuffle(2),
            nn.BatchNorm2d(32), nn.GELU())
        self.up4 = nn.Sequential(
            nn.Conv2d(32, 16 * 4, 3, padding=1), nn.PixelShuffle(2),
            nn.BatchNorm2d(16), nn.GELU())
        self.reduce1 = nn.Sequential(
            nn.Conv2d(128 + dm, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64 + dm, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.reduce3 = nn.Sequential(
            nn.Conv2d(32 + dm, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.out = nn.Conv2d(16, 1, 3, padding=1)
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(128, 1, 1),
            nn.Conv2d(64, 1, 1),
            nn.Conv2d(32, 1, 1),
        ])
        self.refine = EdgeRefinementHead(in_ch=2, h=32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.scale_bias_head = nn.Sequential(
            nn.Flatten(), nn.Linear(dm + 4, 16), nn.ReLU(), nn.Linear(16, 2))
        self.building_detail = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        self.ground_smooth = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, bias=False))

    def forward(self, fused_feats, global_feat, img, rd=None, return_aux=False):
        _, _, H_in, W_in = img.shape
        skip_l2 = fused_feats[1] if len(fused_feats) > 1 else fused_feats[0]
        skip_l1 = fused_feats[0]
        skip_h = fused_feats[0]

        x = self.up1(global_feat)
        skip = F.interpolate(skip_l2, x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce1(x)
        aux1 = self.aux_heads[0](x)

        x = self.up2(x)
        skip = F.interpolate(skip_l1, x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce2(x)
        aux2 = self.aux_heads[1](x)

        x = self.up3(x)
        skip = F.interpolate(skip_h, x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce3(x)
        aux3 = self.aux_heads[2](x)

        x = self.up4(x)
        pred = self.out(x)
        if pred.shape[-2:] != (H_in, W_in):
            pred = F.interpolate(pred, (H_in, W_in), mode='bilinear', align_corners=False)

        gf = self.global_pool(global_feat).flatten(1)
        im = img.mean(dim=(2, 3))[:, :1]
        istd = img.std(dim=(2, 3))[:, :1]
        if rd is None:
            rdm = torch.zeros_like(im)
            rds = torch.zeros_like(im)
        else:
            rdm = rd.mean(dim=(2, 3))[:, :1]
            rds = rd.std(dim=(2, 3))[:, :1]
        si = torch.cat([gf, im, istd, rdm, rds], dim=1)
        sb = self.scale_bias_head(si)
        scale = 1.0 + 0.35 * torch.tanh(sb[:, 0]).view(-1, 1, 1, 1)
        bias = 0.6 * torch.tanh(sb[:, 1]).view(-1, 1, 1, 1)
        pred = scale * pred + bias

        em = image_edge_map(img)
        if self.task == 'building':
            pred = self.refine(pred, em)
            pred = pred + 0.05 * self.building_detail(torch.cat([pred, em], dim=1))
        else:
            low = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
            pred = 0.7 * pred + 0.3 * low
            pred = pred + 0.02 * self.ground_smooth(pred)

        if return_aux:
            return pred, [aux1, aux2, aux3]
        return pred


class MaskSplitModel(nn.Module):
    def __init__(self, dm=64, nh=8):
        super().__init__()
        self.dm = dm
        self.enc = MambaEncoder(6, dm, depths=(3, 4, 3))
        self.cbam = nn.ModuleList([CBAM(dm) for _ in range(3)])
        self.cross_attn = nn.ModuleList([LightCrossAttention(dm, nh) for _ in range(3)])
        self.cross_attn_img = nn.ModuleList([LightCrossAttention(dm, nh) for _ in range(3)])
        self.fusion_gates = nn.ModuleList([DynamicFusionGate(dm) for _ in range(3)])
        self.pos_enc = PositionalEncoding(dm)
        self.global_tf = nn.Sequential(*[LightTransformerBlock(dm, nh, hd=dm * 2) for _ in range(3)])
        self.task_adapt_b = nn.ModuleList([ConvLayer(dm, dm, k=1, p=0) for _ in range(3)])
        self.task_adapt_g = nn.ModuleList([ConvLayer(dm, dm, k=1, p=0) for _ in range(3)])
        self.global_tf_b = nn.Sequential(*[LightTransformerBlock(dm, nh, hd=dm * 2) for _ in range(2)])
        self.global_tf_g = nn.Sequential(*[LightTransformerBlock(dm, nh, hd=dm * 2) for _ in range(2)])
        self.building_decoder = PixelShuffleDecoder(dm, task='building')
        self.ground_decoder = PixelShuffleDecoder(dm, task='ground')
        self.mix_refine = nn.Sequential(
            nn.Conv2d(dm + 1, max(dm // 2, 16), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(dm // 2, 16), 1, 1),
        )

    def forward(self, depth, img, building_mask, reldepth, return_all=False):
        _, _, H, W = img.shape
        bm_bin = (building_mask > 0.5).float()
        bm_soft = F.avg_pool2d(bm_bin, kernel_size=9, stride=1, padding=4)
        rd_detail = reldepth - F.avg_pool2d(reldepth, kernel_size=7, stride=1, padding=3)
        x = torch.cat([depth, img, bm_bin, rd_detail], dim=1)
        feats = self.enc(x)
        feats = [self.cbam[i](f) for i, f in enumerate(feats)]

        fused_feats = []
        for i in range(3):
            d2i = self.cross_attn[i](q=feats[i], k=feats[i], v=feats[i])
            i2d = self.cross_attn_img[i](q=feats[i], k=feats[i], v=feats[i])
            alpha = self.fusion_gates[i](feats[i], d2i, i2d)
            cross_mix = (1.0 - alpha) * d2i + alpha * i2d
            fused_feats.append(0.5 * feats[i] + 0.5 * cross_mix)

        gf = self.pos_enc(fused_feats[2])
        for blk in self.global_tf:
            gf = blk(gf)

        fused_feats_b = []
        fused_feats_g = []
        for i, f in enumerate(fused_feats):
            bm_i = F.interpolate(bm_soft, size=f.shape[-2:], mode='bilinear', align_corners=False)
            fb = self.task_adapt_b[i](f)
            fg = self.task_adapt_g[i](f)
            fused_feats_b.append(fb * (0.05 + 0.95 * bm_i))
            fused_feats_g.append(fg * (0.05 + 0.95 * (1.0 - bm_i)))

        bm_down = F.interpolate(bm_soft, gf.shape[2:], mode='bilinear', align_corners=False)
        building_feat = gf * bm_down
        ground_feat = gf * (1.0 - bm_down)
        for blk in self.global_tf_b:
            building_feat = blk(building_feat)
        for blk in self.global_tf_g:
            ground_feat = blk(ground_feat)

        building_pred, b_aux = self.building_decoder(
            fused_feats_b, building_feat, img, rd=reldepth, return_aux=True)
        ground_pred, g_aux = self.ground_decoder(
            fused_feats_g, ground_feat, img, rd=reldepth, return_aux=True)

        bm_full = F.interpolate(bm_soft, (H, W), mode='bilinear', align_corners=False)
        mix_feat = F.interpolate(fused_feats[0], (H, W), mode='bilinear', align_corners=False)
        mix_delta = self.mix_refine(torch.cat([mix_feat, bm_full], dim=1))
        mix = torch.sigmoid(4.0 * (bm_full - 0.5) + 0.6 * mix_delta)
        pred = mix * building_pred + (1.0 - mix) * ground_pred

        if return_all:
            return pred, building_pred, ground_pred, b_aux, g_aux
        return pred

class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.SmoothL1Loss(reduction='none')
    def forward(self, pred, tgt, mask):
        mask = mask.float()
        pred_safe = torch.where(torch.isnan(pred) | torch.isinf(pred), torch.zeros_like(pred), pred)
        tgt_safe = torch.where(torch.isnan(tgt) | torch.isinf(tgt), torch.zeros_like(tgt), tgt)
        vf = (mask > 0).float()
        n = vf.sum().clamp(min=1)
        loss = self.base(pred_safe, tgt_safe) * vf
        return loss.sum() / n


def edge_loss_fn(pred, tgt, mask):
    mask = mask.float()
    pred_safe = torch.where(torch.isnan(pred) | torch.isinf(pred), torch.zeros_like(pred), pred)
    tgt_safe = torch.where(torch.isnan(tgt) | torch.isinf(tgt), torch.zeros_like(tgt), tgt)
    pgx, pgy = sobel_edges(pred_safe)
    tgx, tgy = sobel_edges(tgt_safe)
    gd = (pgx - tgx).abs() + (pgy - tgy).abs()
    te = (tgx.abs() + tgy.abs()).detach()
    ew = 1.0 + 2.0 * te / (te.mean((-1, -2), True) + 1e-6)
    ew = torch.clamp(ew, 0.0, 10.0)
    if mask.shape != gd.shape:
        mask = F.interpolate(mask, gd.shape[-2:], mode='nearest')
    gd = gd * ew * mask
    return gd.sum() / mask.sum().clamp(min=1)


class SplitLoss(nn.Module):
    def __init__(self, we=0.03, aux_w=0.1, fused_w=0.5, boundary_w=0.15,
                 b_silog_w=0.20, g_smooth_w=0.05, hard_w=0.20, hard_ratio=0.15,
                 b_under_w=0.08, g_under_w=0.04,
                 hard_warmup=0.10, hard_full=0.45, hard_clip=8.0, boundary_k=5):
        super().__init__()
        self.we = we
        self.aux_w = aux_w
        self.fused_w = fused_w
        self.boundary_w = boundary_w
        self.b_silog_w = b_silog_w
        self.g_smooth_w = g_smooth_w
        self.hard_w = hard_w
        self.hard_ratio = hard_ratio
        self.b_under_w = b_under_w
        self.g_under_w = g_under_w
        self.hard_warmup = hard_warmup
        self.hard_full = hard_full
        self.hard_clip = hard_clip
        self.boundary_k = boundary_k
        self.ml = MaskedLoss()

    def si_log_loss(self, pred, tgt, mask):
        v = (mask > 0) & (tgt > 0)
        if v.sum() < 8:
            return pred.new_tensor(0.0)
        p = torch.clamp(pred[v], min=1e-3)
        t = torch.clamp(tgt[v], min=1e-3)
        d = torch.log(p) - torch.log(t)
        return torch.sqrt(torch.clamp((d ** 2).mean() - 0.85 * (d.mean() ** 2), min=1e-8))

    def tv_loss(self, pred, mask):
        gx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
        gy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        vx = gx * mx
        vy = gy * my
        den = (mx.sum() + my.sum()).clamp(min=1)
        return (vx.sum() + vy.sum()) / den

    def hard_loss(self, pred, tgt, mask, ratio=None):
        if ratio is None:
            ratio = self.hard_ratio
        m = (mask > 0)
        if m.sum() < 8:
            return pred.new_tensor(0.0)
        err = F.smooth_l1_loss(pred, tgt, reduction='none')[m]
        err = torch.clamp(err, max=self.hard_clip)
        k = max(1, int(err.numel() * ratio))
        return torch.topk(err, k=k, sorted=False).values.mean()

    def under_loss(self, pred, tgt, mask):
        m = (mask > 0)
        if m.sum() < 8:
            return pred.new_tensor(0.0)
        d = (tgt - pred)[m]
        return F.smooth_l1_loss(torch.relu(d), torch.zeros_like(d))

    def forward(self, pred, building_pred, ground_pred,
                target, valid_mask, building_mask,
                b_aux, g_aux, epoch=0, total_epochs=120):
        bm_bin = (building_mask > 0.5).float()
        bm_mask = valid_mask * bm_bin
        gm_mask = valid_mask * (1.0 - bm_bin)

        building_l1 = self.ml(building_pred, target, bm_mask)
        ground_l1 = self.ml(ground_pred, target, gm_mask)
        fused_loss = self.ml(pred, target, valid_mask)

        build_silog = self.si_log_loss(building_pred, target, bm_mask)
        ground_smooth = self.tv_loss(ground_pred, gm_mask)
        build_under = self.under_loss(building_pred, target, bm_mask)
        ground_under = self.under_loss(ground_pred, target, gm_mask)

        bm_soft = F.avg_pool2d(bm_bin, kernel_size=7, stride=1, padding=3)
        boundary_mask = make_boundary_mask(bm_bin, valid_mask, k=self.boundary_k)
        boundary_loss = 0.5 * (
            self.ml(building_pred, target, boundary_mask) +
            self.ml(ground_pred, target, boundary_mask)
        )

        progress = min((epoch + 1) / max(total_epochs, 1), 1.0)
        if progress <= self.hard_warmup:
            hard_mul = 0.0
        elif progress >= self.hard_full:
            hard_mul = 1.0
        else:
            hard_mul = (progress - self.hard_warmup) / max(self.hard_full - self.hard_warmup, 1e-6)
        hard_ratio_eff = max(0.05, self.hard_ratio * (1.0 - 0.5 * progress))
        hard_w_eff = self.hard_w * hard_mul
        hard_build = self.hard_loss(building_pred, target, bm_mask, ratio=hard_ratio_eff)
        hard_boundary = self.hard_loss(pred, target, boundary_mask, ratio=min(hard_ratio_eff, 0.08))
        el = edge_loss_fn(pred, target, valid_mask)

        aux_loss = torch.tensor(0.0, device=pred.device)
        aux_count = 0
        for aux_pred in b_aux:
            ah, aw = aux_pred.shape[2], aux_pred.shape[3]
            t_r = F.interpolate(target, size=(ah, aw), mode='bilinear', align_corners=False)
            m_r = F.interpolate(bm_mask, size=(ah, aw), mode='nearest')
            aux_loss = aux_loss + self.ml(aux_pred, t_r, m_r)
            aux_count += 1
        for aux_pred in g_aux:
            ah, aw = aux_pred.shape[2], aux_pred.shape[3]
            t_r = F.interpolate(target, size=(ah, aw), mode='bilinear', align_corners=False)
            m_r = F.interpolate(gm_mask, size=(ah, aw), mode='nearest')
            aux_loss = aux_loss + self.ml(aux_pred, t_r, m_r)
            aux_count += 1
        aux_loss = aux_loss / max(aux_count, 1)

        bw = 1.0 + 0.20 * progress

        try:
            bm_count = float(bm_mask.sum().item())
            gm_count = float(gm_mask.sum().item())
        except Exception:
            bm_count = 0.0
            gm_count = 0.0
        if bm_count > 0:
            area_ratio = gm_count / (bm_count + 1e-6)
        else:
            area_ratio = 1.0
        area_scale = min(2.0, max(1.0, math.sqrt(area_ratio)))
        bw = bw * area_scale
        gap_ratio = float((building_l1.detach() / (ground_l1.detach() + 1e-6)).item())
        bw = bw * min(1.4, max(1.0, math.sqrt(gap_ratio)))

        building_total = (
            building_l1 + self.b_silog_w * build_silog + self.b_under_w * build_under +
            hard_w_eff * hard_build
        )
        ground_total = (
            ground_l1 + self.g_smooth_w * ground_smooth + self.g_under_w * ground_under
        )
        boundary_w_eff = self.boundary_w * (0.6 + 0.4 * progress)

        total = (bw * building_total + ground_total + self.fused_w * fused_loss +
                 boundary_w_eff * boundary_loss + self.we * el + self.aux_w * aux_loss +
                 0.5 * hard_w_eff * hard_boundary)

        return total, {
            'building': building_l1.item(),
            'ground': ground_l1.item(),
            'fused': fused_loss.item(),
            'boundary': boundary_loss.item(),
            'b_silog': build_silog.item(),
            'g_smooth': ground_smooth.item(),
            'b_under': build_under.item(),
            'g_under': ground_under.item(),
            'hard_b': hard_build.item(),
            'hard_bd': hard_boundary.item(),
            'edge': el.item(),
            'aux': aux_loss.item(),
            'bw': bw,
            'hw_eff': hard_w_eff,
            'hr_eff': hard_ratio_eff,
            'bd_ratio': float(boundary_mask.sum().item() / valid_mask.sum().clamp(min=1).item()),
            'bm_ratio': float(bm_count / max((bm_count + gm_count), 1e-6)),
        }

class RelDepthAnalyzer:
    def __init__(self):
        self.results = {}

    def analyze(self, loader, max_samples=100):
        patch_spearmans = []
        patch_spearmans_bm = []
        patch_spearmans_gm = []
        height_bin_errors = {k: [] for k in ['0-3m', '3-5m', '5-10m', '10-20m', '20m+']}
        bins = [(0, 3), (3, 5), (5, 10), (10, 20), (20, 9999)]

        with torch.no_grad():
            for bi, b in enumerate(loader):
                if bi >= max_samples:
                    break
                rd = b['rel_depth'].numpy()
                gt = b['depth'].numpy()
                m = b['mask'].numpy()
                bm = b['building_mask'].numpy()

                for i in range(rd.shape[0]):
                    v = m[i, 0] > 0
                    if v.sum() < 10:
                        continue
                    rd_p = rd[i, 0][v]
                    gt_p = gt[i, 0][v]

                    if HAS_SCIPY:
                        sp, _ = scipy_stats.spearmanr(rd_p, gt_p)
                        if not np.isnan(sp):
                            patch_spearmans.append(sp)

                    bm_p = bm[i, 0][v]
                    bm_mask = bm_p > 0.5
                    gm_mask = bm_p <= 0.5

                    if HAS_SCIPY and bm_mask.sum() > 10:
                        sp_bm, _ = scipy_stats.spearmanr(rd_p[bm_mask], gt_p[bm_mask])
                        if not np.isnan(sp_bm):
                            patch_spearmans_bm.append(sp_bm)
                    if HAS_SCIPY and gm_mask.sum() > 10:
                        sp_gm, _ = scipy_stats.spearmanr(rd_p[gm_mask], gt_p[gm_mask])
                        if not np.isnan(sp_gm):
                            patch_spearmans_gm.append(sp_gm)

                    # rel_depth error grouped by GT height bins
                    for (lo, hi), key in zip(bins, height_bin_errors.keys()):
                        in_bin = (gt_p >= lo) & (gt_p < hi)
                        if in_bin.sum() > 0:
                            err = np.abs(rd_p[in_bin] - gt_p[in_bin]).mean()
                            height_bin_errors[key].append(err)

        self.results = {
            'patch_spearman_mean': np.mean(patch_spearmans) if patch_spearmans else 0,
            'patch_spearman_std': np.std(patch_spearmans) if patch_spearmans else 0,
            'patch_spearman_bm': np.mean(patch_spearmans_bm) if patch_spearmans_bm else 0,
            'patch_spearman_gm': np.mean(patch_spearmans_gm) if patch_spearmans_gm else 0,
            'n_patches': len(patch_spearmans),
            'height_bin_errors': {k: np.mean(v) if v else 0 for k, v in height_bin_errors.items()},
            'height_bin_counts': {k: len(v) for k, v in height_bin_errors.items()},
        }
        return self.results
    def print_report(self):
        r = self.results
        print("\n" + "=" * 60)
        print("rel_depth analysis (patch-level Spearman)")
        print("=" * 60)
        print(f"  Patch count: {r['n_patches']}")
        print(f"  Global Spearman: {r['patch_spearman_mean']:.4f} +/- {r['patch_spearman_std']:.4f}")
        print(f"  Building Spearman: {r['patch_spearman_bm']:.4f}")
        print(f"  Ground Spearman: {r['patch_spearman_gm']:.4f}")
        print("\n  rel_depth MAE by height bin (shows scale/shift mismatch):")
        for k in r['height_bin_errors']:
            mae = r['height_bin_errors'][k]
            cnt = r['height_bin_counts'][k]
            print(f"    {k:>8s}: MAE={mae:.3f} ({cnt} patches)")
        print("=" * 60)


    def plot(self, loader, save_path, max_samples=15):
        all_rd, all_gt, all_mask, all_bm = [], [], [], []
        with torch.no_grad():
            for bi, b in enumerate(loader):
                if bi >= max_samples:
                    break
                all_rd.append(b['rel_depth'].numpy())
                all_gt.append(b['depth'].numpy())
                all_mask.append(b['mask'].numpy())
                all_bm.append(b['building_mask'].numpy())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1) Global scatter (sampled)
        rd = np.concatenate(all_rd, 0).flatten()
        gt = np.concatenate(all_gt, 0).flatten()
        m = np.concatenate(all_mask, 0).flatten()
        bm = np.concatenate(all_bm, 0).flatten()
        valid = m > 0
        ax = axes[0, 0]
        n_s = min(10000, int(valid.sum()))
        if n_s > 0:
            idx = np.random.choice(int(valid.sum()), n_s, replace=False)
            ax.scatter(gt[valid][idx], rd[valid][idx], s=1, alpha=0.1)
        ax.set_xlabel('GT Height (m)')
        ax.set_ylabel('rel_depth')
        ax.set_title(f'Global Spearman={self.results["patch_spearman_mean"]:.3f}')

        # 2) Building-region scatter
        ax = axes[0, 1]
        bvalid = valid & (bm > 0.5)
        if bvalid.sum() > 0:
            n_s = min(5000, int(bvalid.sum()))
            idx = np.random.choice(int(bvalid.sum()), n_s, replace=False)
            ax.scatter(gt[bvalid][idx], rd[bvalid][idx], s=1, alpha=0.2, c='red')
        ax.set_xlabel('GT Height (m)')
        ax.set_ylabel('rel_depth')
        ax.set_title(f'Building Spearman={self.results["patch_spearman_bm"]:.3f}')

        # 3) Ground-region scatter
        ax = axes[0, 2]
        gvalid = valid & (bm <= 0.5)
        if gvalid.sum() > 0:
            n_s = min(5000, int(gvalid.sum()))
            idx = np.random.choice(int(gvalid.sum()), n_s, replace=False)
            ax.scatter(gt[gvalid][idx], rd[gvalid][idx], s=1, alpha=0.1, c='blue')
        ax.set_xlabel('GT Height (m)')
        ax.set_ylabel('rel_depth')
        ax.set_title(f'Ground Spearman={self.results["patch_spearman_gm"]:.3f}')

        # 4) MAE by height bins
        ax = axes[1, 0]
        keys = list(self.results['height_bin_errors'].keys())
        maes = [self.results['height_bin_errors'][k] for k in keys]
        ax.bar(keys, maes, color='steelblue', alpha=0.7)
        ax.set_ylabel('MAE (m)')
        ax.set_title('rel_depth MAE by Height')
        ax.tick_params(axis='x', rotation=45)

        # 5-6) Example error heatmaps
        for vis_idx in range(min(2, len(all_rd))):
            ax = axes[1, 1 + vis_idx]
            error = np.abs(all_rd[vis_idx][0] - all_gt[vis_idx][0]) * all_mask[vis_idx][0]
            error = error.squeeze()
            v_max = max(20, error.max())
            im = ax.imshow(error, cmap='hot', vmin=0, vmax=v_max)
            ax.set_title(f'|rel_depth - GT| #{vis_idx}')
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Analysis figure saved: {save_path}")


# =====================================================================
#  Data
# =====================================================================

GPKG_MAP = {
    'NYC': '/root/autodl-tmp/train/nycgpkg.gpkg',
    'LA': '/root/autodl-tmp/train/lagpkg.gpkg',
}

def get_gpkg_for_path(tile_path):
    path_lower = tile_path.lower()
    for city, gpkg in GPKG_MAP.items():
        if city.lower() in path_lower:
            return gpkg
    return None

_gpkg_cache = {}

def load_building_mask_from_gpkg(gpkg_path, tile_geotiff_path):
    if not HAS_GPD or not os.path.exists(gpkg_path):
        return None
    with rasterio.open(tile_geotiff_path) as src:
        tile_crs = src.crs
        tile_transform = src.transform
        tile_h, tile_w = src.height, src.width
    minx = tile_transform.c
    maxy = tile_transform.f
    cache_key = f"{gpkg_path}_{tile_crs}"
    if cache_key not in _gpkg_cache:
        try:
            gdf = gpd.read_file(gpkg_path)
        except Exception:
            return None
        if len(gdf) == 0:
            return None
        gpkg_crs = gdf.crs
        if gpkg_crs is not None and tile_crs is not None:
            try:
                if str(gpkg_crs) != str(tile_crs):
                    gdf = gdf.to_crs(tile_crs)
            except Exception:
                pass
        res_x = abs(tile_transform.a)
        res_y = abs(tile_transform.e)
        all_bounds = gdf.total_bounds
        full_w = int(np.ceil((all_bounds[2] - all_bounds[0]) / res_x))
        full_h = int(np.ceil((all_bounds[3] - all_bounds[1]) / res_y))
        mem_gb = full_w * full_h * 4 / (1024 ** 3)
        if mem_gb > 2.0:
            res_x *= 2; res_y *= 2; full_w //= 2; full_h //= 2
        full_transform = from_bounds(all_bounds[0], all_bounds[1],
                                     all_bounds[2], all_bounds[3],
                                     full_w, full_h)
        shapes = [(geom, 1) for geom in gdf.geometry
                  if geom is not None and not geom.is_empty]
        if not shapes:
            return None
        full_mask = rio_rasterize(shapes, out_shape=(full_h, full_w),
                                  transform=full_transform, fill=0, dtype='uint8')
        _gpkg_cache[cache_key] = {
            'mask': full_mask, 'transform': full_transform,
            'bounds': all_bounds, 'res_x': res_x, 'res_y': res_y}
    cache = _gpkg_cache[cache_key]
    full_mask = cache['mask']
    fb = cache['bounds']
    frx = cache['res_x']
    fry = cache['res_y']
    col_start = int(round((minx - fb[0]) / frx))
    row_start = int(round((fb[3] - maxy) / fry))
    fh, fw = full_mask.shape
    r0 = max(0, row_start)
    r1 = min(fh, row_start + tile_h)
    c0 = max(0, col_start)
    c1 = min(fw, col_start + tile_w)
    if r0 >= r1 or c0 >= c1:
        return np.zeros((tile_h, tile_w), dtype=np.float32)
    patch = full_mask[r0:r1, c0:c1].astype(np.float32)
    result = np.zeros((tile_h, tile_w), dtype=np.float32)
    pr = max(0, -row_start)
    pc = max(0, -col_start)
    ph, pw = patch.shape
    result[pr:pr + ph, pc:pc + pw] = patch
    return result


class DepthDataset(Dataset):
    def __init__(self, doms, gts, rds, bms, transform=None, preload=True, augment=False):
        self.tf = transform
        self.augment = augment
        self.data = []
        if preload:
            print(f"  Preloading {len(doms)} samples...")
            t0 = time.time()
            for i, (d, g, r, bm) in enumerate(zip(doms, gts, rds, bms)):
                try:
                    self.data.append(self._load_one(d, g, r, bm))
                except Exception as e:
                    print(f"    Sample {i} failed: {e}")
                if (i + 1) % 500 == 0:
                    print(f"    Loaded {i + 1}/{len(doms)}")
            print(f"  preload done: {time.time()-t0:.1f}s, samples={len(self.data)}")
            n_bm = sum(1 for s in self.data if s['building_mask'].sum() > 100)
            print(f"  Samples with building pixels: {n_bm}/{len(self.data)}")
            if augment and len(self.data) > 0:
                hs = []
                for s in self.data:
                    r = s['building_mask'].sum() / (s['mask'].sum() + 1e-8)
                    if r > 0.1:
                        for _ in range(min(int(r * 20), 5)):
                            hs.append(s.copy())
                if hs:
                    self.data.extend(hs)
                    print(f"  Oversampling added: +{len(hs)}, total: {len(self.data)}")

    def _read_img(self, p):
        with rasterio.open(p) as s:
            d = s.read()
            if d.dtype != np.uint8:
                if d.dtype == np.uint16:
                    mn, mx = d.min(), d.max()
                    d = ((d.astype(np.float32) - mn) / (mx - mn + 1e-7) * 255).astype(np.uint8)
                else:
                    d = d.astype(np.uint8)
            if s.count == 1:
                d = np.stack([d[0]] * 3, -1)
            else:
                d = d[:3].transpose(1, 2, 0)
                if d.shape[2] == 3:
                    d = d[:, :, ::-1]
        return d

    def _read_f32(self, p):
        with rasterio.open(p) as s:
            return s.read(1).astype(np.float32)

    def _load_one(self, dom_path, gt_path, rd_path, bm_path):
        img = np.zeros((TS, TS, 3), dtype=np.uint8)
        dep = np.zeros((TS, TS), dtype=np.float32)
        rd = np.zeros((TS, TS), dtype=np.float32)

        try:
            img = self._read_img(dom_path)
        except Exception:
            pass

        try:
            dep = self._read_f32(gt_path)
        except Exception:
            pass

        dep = np.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
        dep = np.clip(dep, 0.0, 500.0)
        msk = (~np.isnan(dep)) & (dep != -9999.0) & (dep > 0)
        dep_clean = np.where(msk, dep, 0.0).astype(np.float32)
        mimg = ~np.all(img == 0, 2) if img.ndim == 3 else np.ones(msk.shape, dtype=bool)
        final_mask = (msk & mimg).astype(np.float32)

        building_mask = np.zeros((TS, TS), dtype=np.float32)
        if bm_path and os.path.exists(bm_path):
            try:
                bm = self._read_f32(bm_path)
                building_mask = np.where(np.isnan(bm), 0.0, bm).astype(np.float32)
            except Exception:
                pass
        else:
            gpkg_path = get_gpkg_for_path(dom_path)
            if gpkg_path and os.path.exists(gpkg_path):
                try:
                    bm = load_building_mask_from_gpkg(gpkg_path, dom_path)
                    if bm is not None and bm.sum() > 0:
                        building_mask = bm * final_mask
                except Exception:
                    pass

        building_mask = np.nan_to_num(building_mask, nan=0.0, posinf=0.0, neginf=0.0)
        bm_max = float(building_mask.max()) if building_mask.size > 0 else 0.0
        if bm_max > 1.0:
            if bm_max <= 255.0 + 1e-6:
                building_mask = building_mask / 255.0
            else:
                building_mask = building_mask / (bm_max + 1e-6)
        building_mask = np.clip(building_mask, 0.0, 1.0) * final_mask

        try:
            if rd_path and os.path.exists(rd_path):
                rd = self._read_f32(rd_path)
                rd = np.nan_to_num(rd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception:
            pass

        v = final_mask > 0
        if v.any():
            rd_v = rd[v]
            q_lo = np.percentile(rd_v, 2.0)
            q_hi = np.percentile(rd_v, 98.0)
            den = max(q_hi - q_lo, 1e-4)
            rd = (rd - q_lo) / den
            rd = np.clip(rd, 0.0, 1.0)
        else:
            rd = np.zeros_like(rd, dtype=np.float32)
        rd = rd.astype(np.float32) * final_mask

        return {
            'image': img, 'rel_depth': rd, 'depth': dep_clean,
            'mask': final_mask, 'building_mask': building_mask,
        }


    def __len__(self):
        return len(self.data)

    def _augment(self, s):
        keys = ['image', 'rel_depth', 'depth', 'mask', 'building_mask']
        copies = {k: s[k].copy() for k in keys}
        if np.random.rand() < 0.5:
            copies = {k: np.flip(v, axis=1).copy() for k, v in copies.items()}
        if np.random.rand() < 0.5:
            copies = {k: np.flip(v, axis=0).copy() for k, v in copies.items()}
        k = np.random.randint(0, 4)
        if k > 0:
            copies = {k2: np.rot90(v, k).copy() for k2, v in copies.items()}
        for k2 in keys:
            s[k2] = copies[k2]
        return s

    def __getitem__(self, idx):
        s = self.data[idx].copy()
        if self.augment:
            s = self._augment(s)
        if self.tf:
            s = self.tf(s)
        r = {}
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                r[k] = v
            elif isinstance(v, np.ndarray):
                r[k] = torch.from_numpy(v.astype(np.float32))
            else:
                r[k] = v
        return r
class FastToTensor:
    def __call__(self, s):
        if isinstance(s['image'], np.ndarray):
            img = s['image'].astype(np.float32) / 255.0
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            s['image'] = torch.from_numpy(img)
        for k in ('rel_depth', 'depth', 'mask', 'building_mask'):
            if k in s and isinstance(s[k], np.ndarray):
                s[k] = torch.from_numpy(s[k].astype(np.float32)).unsqueeze(0)
        return s

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class FastNorm:
    def __init__(self, m, st):
        self.m = torch.tensor(m).view(-1, 1, 1)
        self.s = torch.tensor(st).view(-1, 1, 1)
    def __call__(self, s):
        if 'image' in s:
            s['image'] = (s['image'] - self.m) / self.s
        return s


def make_loader(bs, doms, gts, rds, bms, train=True, nw=0):
    tf = transforms.Compose([FastToTensor(), FastNorm(MEAN, STD)])
    ds = DepthDataset(doms, gts, rds, bms, transform=tf, preload=True, augment=train)
    print(f"  Dataset: {len(ds)}")
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=nw, pin_memory=True, drop_last=train)


def collect_data(args):
    all_d, all_g, all_r, all_b = [], [], [], []
    print("=" * 50)
    print("Loading city data (512 tiles)...")
    for cb in args.city_roots:
        cn = os.path.basename(cb.rstrip('/'))
        if not os.path.exists(cb):
            continue
        found_txt = None
        for txt_name in ['train_512.txt', 'val_512.txt', 'train.txt', 'val.txt', 'test.txt', 'all.txt']:
            p = os.path.join(cb, txt_name)
            if os.path.exists(p):
                found_txt = p
                break
        if found_txt is None:
            print(f"  {cn}: index txt not found")
            continue
        count = 0
        with open(found_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                dom_rel = parts[0].replace('\\', '/')
                if len(parts) >= 3:
                    gt_rel = parts[1].replace('\\', '/')
                    rd_rel = parts[2].replace('\\', '/')
                else:
                    gt_rel = parts[1].replace('\\', '/')
                    rd_rel = None
                fn = os.path.basename(dom_rel)
                dom_abs = os.path.join(cb, dom_rel)
                gt_abs = os.path.join(cb, gt_rel)
                if rd_rel is not None:
                    rd_abs = os.path.join(cb, rd_rel)
                else:
                    rd_abs = os.path.join(cb, 'reldepth_512', fn)
                    if not os.path.exists(rd_abs):
                        rd_abs = os.path.join(cb, 'reldepth', fn)
                bm_abs = os.path.join(cb, 'building_mask_512', fn)
                if not os.path.exists(bm_abs):
                    bm_abs = ""
                all_d.append(dom_abs)
                all_g.append(gt_abs)
                all_r.append(rd_abs if os.path.exists(rd_abs) else "")
                all_b.append(bm_abs)
                count += 1
        print(f"  {cn}/{os.path.basename(found_txt)}: {count}")
    vd, vg, vr, vb = [], [], [], []
    for d, g, r, b in zip(all_d, all_g, all_r, all_b):
        if os.path.exists(d) and os.path.exists(g):
            vd.append(d); vg.append(g); vr.append(r); vb.append(b)
    print(f"\nValid samples: {len(vd)} / {len(all_d)}")
    return vd, vg, vr, vb

'''python main.py \
    --city_roots /root/autodl-tmp/train/NYC /root/autodl-tmp/train/LA \
    --batch_size 20 \
    --epoch 300 \
    --lr 1e-4 \
    --gc 0.5 \
    --aux_w 0.2 \
    --we 0.08 \
    --nw 4 \
    --skip_bm'''

# =====================================================================
#  Train / Validate
# =====================================================================

def train_epoch(model, loader, opt, crit, dev, gc_val=0.5,
                epoch=0, total_epochs=120, accum_steps=4):
    model.train()
    tl, ns = 0.0, 0
    skipped = 0
    info_acc = {}
    opt.zero_grad(set_to_none=True)
    for bi, b in enumerate(loader):
        rd = b['rel_depth'].to(dev, non_blocking=True)
        im = b['image'].to(dev, non_blocking=True)
        t = b['depth'].to(dev, non_blocking=True)
        m = b['mask'].to(dev, non_blocking=True)
        bm = b['building_mask'].to(dev, non_blocking=True)

        if torch.isnan(rd).any() or torch.isinf(rd).any():
            skipped += 1; continue
        if torch.isnan(im).any() or torch.isinf(im).any():
            skipped += 1; continue
        if torch.isnan(t).any() or torch.isinf(t).any():
            skipped += 1; continue

        pred, bp, gp, baux, gaux = model(rd, im, bm, reldepth=rd, return_all=True)
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            skipped += 1; continue

        loss, info = crit(pred, bp, gp, t, m, bm, baux, gaux,
                          epoch=epoch, total_epochs=total_epochs)
        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1; continue

        loss = loss / accum_steps
        loss.backward()

        tl += loss.item() * accum_steps * rd.size(0)
        ns += rd.size(0)
        for k, v in info.items():
            info_acc[k] = info_acc.get(k, 0.0) + v

        if (bi + 1) % accum_steps == 0:
            bad = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        bad = True; break
            if bad:
                opt.zero_grad(set_to_none=True)
                skipped += 1; continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), gc_val)
            opt.step()
            opt.zero_grad(set_to_none=True)



    if skipped > 0:
        print(f"    Skipped {skipped} batches")
    nb = max(len(loader) - skipped, 1)
    avg = {k: v / nb for k, v in info_acc.items()}
    return tl / max(ns, 1), avg


def validate(model, loader, crit, dev, epoch=0, total_epochs=120):
    def _corr(a, b):
        if a.numel() < 16:
            return float('nan')
        a = a.float(); b = b.float()
        a = a - a.mean(); b = b - b.mean()
        den = torch.sqrt((a * a).mean() * (b * b).mean()) + 1e-6
        return ((a * b).mean() / den).item()

    model.eval()
    tl, ns = 0.0, 0
    all_abs, all_sq, all_pix = 0.0, 0.0, 0
    bm_abs, bm_sq, bm_pix = 0.0, 0.0, 0
    gm_abs, gm_sq, gm_pix = 0.0, 0.0, 0
    bd_abs, bd_pix = 0.0, 0
    bm_on_gm_abs, bm_on_gm_pix = 0.0, 0
    gm_on_bm_abs, gm_on_bm_pix = 0.0, 0
    gm_pred_mean_sum, gm_pred_mean_n = 0.0, 0
    b_bias_sum, b_bias_n = 0.0, 0
    g_bias_sum, g_bias_n = 0.0, 0
    rd_corr_all, rd_corr_all_n = 0.0, 0
    rd_corr_b, rd_corr_b_n = 0.0, 0
    rd_corr_g, rd_corr_g_n = 0.0, 0
    tgt_std_b, tgt_std_b_n = 0.0, 0
    tgt_std_g, tgt_std_g_n = 0.0, 0
    hard_b_sum, hard_b_n = 0.0, 0
    hard_g_sum, hard_g_n = 0.0, 0

    with torch.no_grad():
        for bi, b in enumerate(loader):
            rd = b['rel_depth'].to(dev, non_blocking=True)
            im = b['image'].to(dev, non_blocking=True)
            t = b['depth'].to(dev, non_blocking=True)
            m = b['mask'].to(dev, non_blocking=True)
            bm = b['building_mask'].to(dev, non_blocking=True)

            pred, bp, gp, baux, gaux = model(rd, im, bm, reldepth=rd, return_all=True)
            loss, _ = crit(pred, bp, gp, t, m, bm, baux, gaux,
                           epoch=epoch, total_epochs=total_epochs)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            tl += loss.item() * rd.size(0)
            ns += rd.size(0)

            for i in range(rd.size(0)):
                v2d = (m[i, 0] > 0) if m.dim() == 4 else (m[i] > 0)
                if not v2d.any():
                    continue
                pi = pred[i, 0] if pred.dim() == 4 else pred[i]
                ti = t[i, 0] if t.dim() == 4 else t[i]
                rdi = rd[i, 0] if rd.dim() == 4 else rd[i]
                bmi = bm[i, 0] if bm.dim() == 4 else bm[i]
                bpi = bp[i, 0] if bp.dim() == 4 else bp[i]
                gpi = gp[i, 0] if gp.dim() == 4 else gp[i]

                err = (pi[v2d] - ti[v2d])
                all_abs += err.abs().sum().item()
                all_sq += (err ** 2).sum().item()
                all_pix += int(v2d.sum().item())

                bmask = v2d & (bmi > 0.5)
                gmask = v2d & (bmi <= 0.5)
                bd2d = make_boundary_mask(
                    (bmi > 0.5).float().unsqueeze(0).unsqueeze(0),
                    v2d.float().unsqueeze(0).unsqueeze(0),
                    k=max(3, int(getattr(crit, "boundary_k", 5)))
                )[0, 0] > 0
                bdmask = v2d & bd2d

                if bmask.any():
                    berr = (pi[bmask] - ti[bmask])
                    bm_abs += berr.abs().sum().item()
                    bm_sq += (berr ** 2).sum().item()
                    bm_pix += int(bmask.sum().item())
                    b_bias_sum += berr.mean().item(); b_bias_n += 1
                    tgt_std_b += ti[bmask].std().item(); tgt_std_b_n += 1
                    k = max(1, int(berr.numel() * 0.10))
                    hard_b_sum += torch.topk(berr.abs(), k=k, sorted=False).values.mean().item()
                    hard_b_n += 1
                if gmask.any():
                    gerr = (pi[gmask] - ti[gmask])
                    gm_abs += gerr.abs().sum().item()
                    gm_sq += (gerr ** 2).sum().item()
                    gm_pix += int(gmask.sum().item())
                    g_bias_sum += gerr.mean().item(); g_bias_n += 1
                    tgt_std_g += ti[gmask].std().item(); tgt_std_g_n += 1
                    k = max(1, int(gerr.numel() * 0.10))
                    hard_g_sum += torch.topk(gerr.abs(), k=k, sorted=False).values.mean().item()
                    hard_g_n += 1
                if bdmask.any():
                    bd_abs += (pi[bdmask] - ti[bdmask]).abs().sum().item()
                    bd_pix += int(bdmask.sum().item())

                if gmask.any():
                    e = (bpi[gmask] - ti[gmask]).abs()
                    bm_on_gm_abs += e.sum().item()
                    bm_on_gm_pix += int(gmask.sum().item())
                    gm_pred_mean_sum += gpi[gmask].abs().mean().item()
                    gm_pred_mean_n += 1
                if bmask.any():
                    e = (gpi[bmask] - ti[bmask]).abs()
                    gm_on_bm_abs += e.sum().item()
                    gm_on_bm_pix += int(bmask.sum().item())

                c = _corr(rdi[v2d], ti[v2d])
                if not np.isnan(c):
                    rd_corr_all += c; rd_corr_all_n += 1
                if bmask.any():
                    c = _corr(rdi[bmask], ti[bmask])
                    if not np.isnan(c):
                        rd_corr_b += c; rd_corr_b_n += 1
                if gmask.any():
                    c = _corr(rdi[gmask], ti[gmask])
                    if not np.isnan(c):
                        rd_corr_g += c; rd_corr_g_n += 1

    bm_mae = bm_abs / max(bm_pix, 1)
    gm_mae = gm_abs / max(gm_pix, 1)
    bd_mae = bd_abs / max(bd_pix, 1)

    if epoch % 10 == 0 or epoch == 1:
        print(f"\n  [DIAG epoch {epoch}]")
        print(f"  Building MAE={bm_mae:.4f} (pix={bm_pix})")
        print(f"  Ground   MAE={gm_mae:.4f} (pix={gm_pix})")
        print(f"  Boundary MAE={bd_mae:.4f} (pix={bd_pix})")
        print(f"  Pixel ratio BM={bm_pix/max(all_pix,1):.3f} GM={gm_pix/max(all_pix,1):.3f} BD={bd_pix/max(all_pix,1):.3f}")
        print(f"  MAE gap (B-G)={bm_mae-gm_mae:.4f}")
        print(f"  Hard10% MAE B={hard_b_sum/max(hard_b_n,1):.4f} G={hard_g_sum/max(hard_g_n,1):.4f}")
        print(f"  B-branch on Ground={bm_on_gm_abs/max(bm_on_gm_pix,1):.4f}")
        print(f"  G-branch on Building={gm_on_bm_abs/max(gm_on_bm_pix,1):.4f}")
        print(f"  Branch bias B={b_bias_sum/max(b_bias_n,1):.4f} G={g_bias_sum/max(g_bias_n,1):.4f}")
        print(f"  rel_depth corr all={rd_corr_all/max(rd_corr_all_n,1):.4f} B={rd_corr_b/max(rd_corr_b_n,1):.4f} G={rd_corr_g/max(rd_corr_g_n,1):.4f}")
        print(f"  target std B={tgt_std_b/max(tgt_std_b_n,1):.4f} G={tgt_std_g/max(tgt_std_g_n,1):.4f}")
        print(f"  G-pred mean(|pred|) on ground={gm_pred_mean_sum/max(gm_pred_mean_n,1):.4f}")

    return {
        'loss': tl / max(ns, 1),
        'mae': all_abs / max(all_pix, 1),
        'rmse': math.sqrt(all_sq / max(all_pix, 1)),
        'bm_mae': bm_mae,
        'gm_mae': gm_mae,
        'bm_n': bm_pix, 'gm_n': gm_pix,
    }
def plot_all(tr, vr, save):
    ep = range(1, len(tr) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(ep, tr, 'b', label='Train')
    ax[0, 0].plot(ep, [v['loss'] for v in vr], 'r', label='Val')
    ax[0, 0].set(title='Loss', xlabel='Epoch')
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(ep, [v['mae'] for v in vr], 'g', label='MAE')
    ax[0, 1].plot(ep, [v['rmse'] for v in vr], 'orange', label='RMSE')
    ax[0, 1].set(title='Overall', xlabel='Epoch')
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    ax[1, 0].plot(ep, [v['bm_mae'] for v in vr], 'r', label='Building MAE')
    ax[1, 0].plot(ep, [v['gm_mae'] for v in vr], 'b', label='Ground MAE')
    ax[1, 0].set(title='MAE by Mask', xlabel='Epoch')
    ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    ax[1, 1].axis('off')
    ax[1, 1].text(0.1, 0.5,
                  f"MAE={vr[-1]['mae']:.4f}\nRMSE={vr[-1]['rmse']:.4f}\n"
                  f"Building MAE={vr[-1]['bm_mae']:.4f}\n"
                  f"Ground MAE={vr[-1]['gm_mae']:.4f}",
                  fontsize=12, verticalalignment='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save, 'curves.png'), dpi=150)
    plt.close()


def generate_building_masks(city_roots):
    for cb in city_roots:
        cn = os.path.basename(cb.rstrip('/'))
        gpkg_path = get_gpkg_for_path(cb)
        if gpkg_path is None or not os.path.exists(gpkg_path):
            print(f"  {cn}: missing GPKG, skip")
            continue
        dom_dir = os.path.join(cb, 'dom_512')
        bm_dir = os.path.join(cb, 'building_mask_512')
        os.makedirs(bm_dir, exist_ok=True)
        if not os.path.exists(dom_dir):
            print(f"  {cn}: dom_512 not found")
            continue
        dom_files = sorted([f for f in os.listdir(dom_dir) if f.endswith('.tif')])
        print(f"  {cn}: building mask ({len(dom_files)} tiles)")
        for fn in dom_files:
            save_path = os.path.join(bm_dir, fn)
            if os.path.exists(save_path):
                continue
            try:
                bm = load_building_mask_from_gpkg(gpkg_path, os.path.join(dom_dir, fn))
                if bm is not None and bm.sum() > 0:
                    with rasterio.open(os.path.join(dom_dir, fn)) as src:
                        profile = src.profile.copy()
                    profile.update(dtype=rasterio.float32, count=1)
                    with rasterio.open(save_path, 'w', **profile) as dst:
                        dst.write(bm.astype(np.float32), 1)
            except Exception:
                pass
        print(f"  {cn}: done")


# =====================================================================
#  Main
# =====================================================================

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--city_roots', nargs='+',
                    default=['/root/autodl-tmp/train/NYC', '/root/autodl-tmp/train/LA'])
    pa.add_argument('--dm', type=int, default=64)
    pa.add_argument('--nh', type=int, default=8)
    pa.add_argument('--we', type=float, default=0.03)
    pa.add_argument('--aux_w', type=float, default=0.1)
    pa.add_argument('--fused_w', type=float, default=0.5)
    pa.add_argument('--boundary_w', type=float, default=0.15)
    pa.add_argument('--b_silog_w', type=float, default=0.20)
    pa.add_argument('--g_smooth_w', type=float, default=0.05)
    pa.add_argument('--hard_w', type=float, default=0.20)
    pa.add_argument('--hard_ratio', type=float, default=0.15)
    pa.add_argument('--b_under_w', type=float, default=0.08)
    pa.add_argument('--g_under_w', type=float, default=0.04)
    pa.add_argument('--hard_warmup', type=float, default=0.10)
    pa.add_argument('--hard_full', type=float, default=0.45)
    pa.add_argument('--hard_clip', type=float, default=8.0)
    pa.add_argument('--boundary_k', type=int, default=5)
    pa.add_argument('--epoch', type=int, default=300)
    pa.add_argument('--batch_size', type=int, default=4)
    pa.add_argument('--lr', type=float, default=1e-4)
    pa.add_argument('--gc', type=float, default=0.5)
    pa.add_argument('--nw', type=int, default=0)
    pa.add_argument('--save_int', type=int, default=10)
    pa.add_argument('--skip_bm', action='store_true')
    pa.add_argument('--analyze_only', action='store_true')
    args = pa.parse_args()

    print(f"Input: 6ch | Size: {TS}x{TS} | dm: {args.dm} | nh: {args.nh}")

    if not args.skip_bm:
        print("\nBuilding mask...")
        generate_building_masks(args.city_roots)

    all_d, all_g, all_r, all_b = collect_data(args)
    if len(all_d) == 0:
        print("no data"); return

    N = len(all_d)
    idx = np.arange(N); np.random.seed(42); np.random.shuffle(idx)
    ts = int(N * 0.8)
    td = [all_d[i] for i in idx[:ts]]; tg = [all_g[i] for i in idx[:ts]]
    tr_ = [all_r[i] for i in idx[:ts]]; tb_ = [all_b[i] for i in idx[:ts]]
    vd = [all_d[i] for i in idx[ts:]]; vg = [all_g[i] for i in idx[ts:]]
    vr_ = [all_r[i] for i in idx[ts:]]; vb_ = [all_b[i] for i in idx[ts:]]
    print(f"Train: {len(td)} | Val: {len(vd)}")

    print(f"\nDataLoader (bs={args.batch_size})...")
    tl = make_loader(args.batch_size, td, tg, tr_, tb_, True, args.nw)
    vl = make_loader(args.batch_size, vd, vg, vr_, vb_, False, args.nw)
    if tl is None or vl is None:
        print("DataLoader failed"); return
    print(f"Train batches: {len(tl)} | Val batches: {len(vl)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    analyzer = RelDepthAnalyzer()
    analyzer.analyze(vl, max_samples=100)
    analyzer.print_report()

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sp = os.path.join('./checkpoints_v13a', ts_str)
    os.makedirs(sp, exist_ok=True)
    analyzer.plot(vl, os.path.join(sp, 'reldepth_analysis.png'))

    if args.analyze_only:
        print("analysis done, exit"); return

    model = MaskSplitModel(args.dm, args.nh).to(dev)
    tp = sum(p.numel() for p in model.parameters())
    print(f"Params: {tp:,}")

    crit = SplitLoss(
        args.we, args.aux_w, args.fused_w, args.boundary_w,
        args.b_silog_w, args.g_smooth_w, args.hard_w, args.hard_ratio,
        args.b_under_w, args.g_under_w, args.hard_warmup, args.hard_full,
        args.hard_clip, args.boundary_k
    ).to(dev)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epoch - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n{'='*60}")
    print("Train v14 split-task model")
    print(f"  Save: {sp}")
    print(f"  Params: {tp:,}")
    print(f"{'='*60}\n")

    bv = float('inf')
    tr_ls, vr_ls = [], []

    for epoch in range(args.epoch):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n--- Epoch {epoch+1}/{args.epoch} LR={lr_now:.2e} ---")

        tl_, info = train_epoch(model, tl, opt, crit, dev, args.gc,
                                epoch=epoch, total_epochs=args.epoch)
        tr_ls.append(tl_)
        scheduler.step()

        vm = validate(model, vl, crit, dev, epoch=epoch+1, total_epochs=args.epoch)
        vr_ls.append(vm)

        print(f"  Train: {tl_:.4f} "
              f"(b={info.get('building',0):.3f} g={info.get('ground',0):.3f} "
              f"f={info.get('fused',0):.3f} bd={info.get('boundary',0):.3f} "
              f"si={info.get('b_silog',0):.3f} gs={info.get('g_smooth',0):.3f} "
              f"ub={info.get('b_under',0):.3f} ug={info.get('g_under',0):.3f} "
              f"hb={info.get('hard_b',0):.3f} hbd={info.get('hard_bd',0):.3f} "
              f"hw={info.get('hw_eff',0):.3f} hr={info.get('hr_eff',0):.3f} "
              f"bmr={info.get('bm_ratio',0):.3f} bdr={info.get('bd_ratio',0):.3f})")
        print(f"  Val:   loss={vm['loss']:.4f} MAE={vm['mae']:.4f} RMSE={vm['rmse']:.4f}")
        print(f"         Building: MAE={vm['bm_mae']:.4f} ({vm['bm_n']})")
        print(f"         Ground:   MAE={vm['gm_mae']:.4f} ({vm['gm_n']})")

        if vm['loss'] < bv:
            bv = vm['loss']
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': bv},
                       os.path.join(sp, 'best.pth'))
            print(f"  * Best: {bv:.4f}")

        if (epoch + 1) % args.save_int == 0:
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': vm['loss']},
                       os.path.join(sp, f'ckpt_{epoch+1}.pth'))

        if (epoch + 1) % 10 == 0 or epoch == args.epoch - 1:
            plot_all(tr_ls, vr_ls, sp)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({'epoch': args.epoch, 'model_state_dict': model.state_dict(),
                'loss': vr_ls[-1]['loss']}, os.path.join(sp, 'final.pth'))

    with open(os.path.join(sp, 'log.csv'), 'w') as f:
        f.write("ep,train,val,mae,rmse,bm_mae,gm_mae,bm_n,gm_n\n")
        for i, (t, v) in enumerate(zip(tr_ls, vr_ls)):
            f.write(f"{i+1},{t:.6f},{v['loss']:.6f},{v['mae']:.6f},"
                    f"{v['rmse']:.6f},{v['bm_mae']:.6f},{v['gm_mae']:.6f},"
                    f"{v['bm_n']},{v['gm_n']}\n")

    plot_all(tr_ls, vr_ls, sp)
    print(f"\nDone! Best={bv:.4f} | {sp}")


if __name__ == "__main__":
    main()








