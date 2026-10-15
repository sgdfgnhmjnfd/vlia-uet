import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_root = Path('/media/dhqg/d/datasets/egointent/cache/temporal_intention_v1')
why_pca_p = Path('/media/dhqg/d/vlia_outputs/stage_a_clean_v1/why_pca_960_to_256.pt')
what_pca_p = Path('/media/dhqg/d/vlia_outputs/structured_intention_v1/what_pca_960_to_256.pt')

def get_pca(p):
    obj = torch.load(p, map_location='cpu')
    mean = obj.get('mean', torch.zeros(960)).squeeze().float()
    comp = obj['components'].squeeze().float()
    basis = comp if comp.shape == (960, 256) else comp.T
    return lambda x: (x.float() - mean) @ basis

why_tf = get_pca(why_pca_p)
what_tf = get_pca(what_pca_p)

def load_data(split):
    rows = []
    for p in sorted((temp_root / split).glob('*.pt')):
        x = torch.load(p, map_location='cpu')
        rows.append({
            'frames': x['frame_features'].float(),
            'why': x['why'],
            'what': x.get('what', ''),
            'task': x['task'],
            'why_feat': F.normalize(why_tf(x['why_feature'].reshape(1, 960)).squeeze(0), dim=-1),
            'what_feat': F.normalize(what_tf(x['what_feature'].reshape(1, 960)).squeeze(0), dim=-1),
        })
    return rows

train_r = load_data('train')
val_r = load_data('val')

train_frames = torch.stack([r['frames'] for r in train_r]).to(device)
val_frames = torch.stack([r['frames'] for r in val_r]).to(device)

train_why = torch.stack([r['why_feat'] for r in train_r]).to(device)
val_why = torch.stack([r['why_feat'] for r in val_r]).to(device)

train_what = torch.stack([r['what_feat'] for r in train_r]).to(device)
val_what = torch.stack([r['what_feat'] for r in val_r]).to(device)

def build_mask(labels):
    n = len(labels)
    m = torch.zeros((n, n), dtype=torch.bool, device=device)
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    for idxs in groups.values():
        idx = torch.tensor(idxs, device=device)
        m[idx[:, None], idx[None, :]] = True
    return m

why_mask = build_mask([r['why'] for r in train_r])
what_mask = build_mask([r['what'] for r in train_r])
task_mask = build_mask([r['task'] for r in train_r])

def infonce(pred, target, mask, temp=0.07):
    logits = (pred @ target.T) / temp
    log_p = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = mask.sum(dim=1).clamp_min(1)
    return -(log_p * mask.float()).sum(dim=1).div(counts).mean()

def cos_loss(pred, target):
    return (1.0 - (pred * target).sum(dim=-1)).mean()

def same_task_hardneg(pred, target, pos_mask, t_mask, margin=0.10):
    sim = pred @ target.T
    neg_mask = t_mask & (~pos_mask)
    pos_sim = sim.masked_fill(~pos_mask, float('-inf')).max(dim=1).values
    neg_sim = sim.masked_fill(~neg_mask, float('-inf')).max(dim=1).values
    valid = torch.isfinite(pos_sim) & torch.isfinite(neg_sim)
    if not valid.any():
        return torch.zeros((), device=device)
    return F.relu(margin + neg_sim[valid] - pos_sim[valid]).mean()

def eval_mrr(pred, target, labels):
    sim = pred @ target.T
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    ranks = []
    for i, l in enumerate(labels):
        targets = set(groups[l])
        sorted_j = torch.argsort(sim[i], descending=True).tolist()
        for r, j in enumerate(sorted_j, start=1):
            if j in targets:
                ranks.append(r)
                break
    rt = torch.tensor(ranks, dtype=torch.float32)
    return (1.0 / rt).mean().item(), (rt <= 1).float().mean().item()

class LightweightPool(nn.Module):
    def __init__(self, in_dim=960, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1, bias=False)
        )
    def forward(self, x):
        scores = self.net(x).squeeze(-1) # (B, T)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * attn).sum(dim=1)

class Model(nn.Module):
    def __init__(self, mode="light_guided"):
        super().__init__()
        self.mode = mode
        self.pool = LightweightPool()
        self.why_head = nn.Sequential(
            nn.Linear(960, 512), nn.GELU(), nn.Linear(512, 256), nn.LayerNorm(256)
        )
        self.what_head = nn.Sequential(
            nn.Linear(960, 512), nn.GELU(), nn.Linear(512, 256), nn.LayerNorm(256)
        )
        self.visual_state = nn.Sequential(
            nn.Linear(960, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.reasoner = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256), nn.LayerNorm(256)
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 256), nn.Sigmoid()
        )

    def forward(self, frames, oracle_what=None):
        if self.mode == "mean_pool_direct":
            vis = frames.mean(dim=1)
            why = F.normalize(self.why_head(vis), dim=-1)
            return why, None, None

        vis = self.pool(frames)
        direct_why = F.normalize(self.why_head(vis), dim=-1)

        if self.mode == "light_direct":
            return direct_why, None, None

        pred_what = F.normalize(self.what_head(vis), dim=-1)
        what_used = oracle_what if (self.mode == "oracle" and oracle_what is not None) else pred_what

        v_state = self.visual_state(vis)
        fused = torch.cat([v_state, what_used], dim=-1)
        res = F.normalize(self.reasoner(fused), dim=-1)
        g = self.gate(fused)
        why = F.normalize(direct_why + g * res, dim=-1)
        return why, pred_what, g

modes = ["mean_pool_direct", "light_direct", "light_guided", "light_hardneg", "oracle"]

for mode in modes:
    m = Model(mode=mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best_mrr = 0.0
    best_top1 = 0.0
    best_ep = 0

    for ep in range(1, 81):
        m.train()
        opt.zero_grad()
        oracle_w = train_what if mode == "oracle" else None
        why_p, what_p, g = m(train_frames, oracle_w)

        loss_why = infonce(why_p, train_why, why_mask) + 0.1 * cos_loss(why_p, train_why)
        loss = loss_why

        if what_p is not None:
            loss_what = infonce(what_p, train_what, what_mask) + 0.1 * cos_loss(what_p, train_what)
            if mode == "light_hardneg":
                loss_what += 0.3 * same_task_hardneg(what_p, train_what, what_mask, task_mask)
            loss += 0.3 * loss_what

        loss.backward()
        opt.step()

        m.eval()
        with torch.no_grad():
            v_oracle = val_what if mode == "oracle" else None
            v_why, v_what, _ = m(val_frames, v_oracle)
            mrr, top1 = eval_mrr(v_why, val_why, [r['why'] for r in val_r])
            if mrr > best_mrr:
                best_mrr = mrr
                best_top1 = top1
                best_ep = ep

    print(f"[{mode:18s}] Best Ep: {best_ep:2d} | WHY MRR: {best_mrr:.6f} | Top1: {best_top1:.6f}")
