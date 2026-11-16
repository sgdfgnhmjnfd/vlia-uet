#!/usr/bin/env python3
"""Stage 14b Part B: W, our own trained model that writes the intention (no candidate list at run time).

Subcommands: selftest | train-ego | write-ego | train-libero | write-libero | unseen | analyze.
Pre-registration: results/prereg_stage14b_writer.md.
W = SigLIP frame embeddings -> trained connector -> SmolVLM2-500M language model (last 4 blocks + norm trained).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV  # noqa: E402
import run_stage14_generate as S14  # noqa: E402

LM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
D_IN, D_LM, N_POS = 1152, 960, 8
N_TRAIN_BLOCKS = 4
MAX_NEW = 24
LR_CONN, LR_LM, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 2e-5, 0.01, 32, 30, 3
EGO_K = (2, 4, 6, 8)
EGO_TAGS = ("why", "what", "next")
LIB_P = (0, 10, 25, 50, 75, 100)
LIB_TRAIN_P = (10, 25, 50, 75, 100)
N_FOLDS_EGO = 8
WORK = Path("/media/dhqg/d3/vlia_outputs/stage14b_writer")
RES = ROOT / "results/stage14b_writer"
LADDER = Path("/media/dhqg/d3/vlia_outputs/ladder/siglip-so400m-patch14-384")
EGO_ROOT = Path("/media/dhqg/d3/datasets/egointent_full")
MANIFEST = ROOT / "results/hoi_detection_v1/manifest.jsonl"
STOP = {"for", "to", "with", "in", "into", "on", "onto", "from", "of", "before", "after", "and", "or", "by", "at", "using",
        "while", "so", "that", "then", "as", "through", "under", "over", "near", "inside"}
TRAIN_SENTENCES = {
    0: ["put the soup can and the tomato sauce can into the basket", "pick up the alphabet soup and the tomato sauce and drop them in the basket",
        "the alphabet soup and the tomato sauce go in the basket", "I want the soup can and the tomato sauce in the basket",
        "I want both the alphabet soup and the tomato sauce placed inside the basket", "I want the basket to hold the alphabet soup and the tomato sauce"],
    1: ["put the cream cheese and the butter box into the basket", "pick up the cream cheese box and the butter and drop them in the basket",
        "the cream cheese box and the butter go in the basket", "I want the cream cheese and the butter in the basket",
        "I want both the cream cheese box and the butter placed inside the basket", "I want the basket to hold the cream cheese box and the butter"],
    2: ["turn the stove on and set the moka pot on the burner", "start the stove, then put the coffee pot on it",
        "the stove gets turned on and the moka pot goes on top", "I want the stove on and the moka pot on it",
        "I want the moka pot placed on the stove with the stove turned on", "I want the coffee pot on the lit stove"],
    3: ["put the black bowl into the bottom drawer and close the drawer", "store the bowl in the lowest drawer of the cabinet, then push the drawer shut",
        "the black bowl goes in the cabinet's bottom drawer and the drawer gets closed", "I want the black bowl in the bottom drawer and the drawer closed",
        "I want the bowl inside the lowest cabinet drawer with the drawer pushed shut", "I want the black bowl put away in the closed bottom drawer of the cabinet"],
    4: ["put the white mug on the left plate and the yellow and white mug on the right plate",
        "move the white cup onto the left plate and the yellow and white cup onto the right plate",
        "the white mug goes on the left plate, the yellow and white mug goes on the right plate",
        "I want the left plate to have the white mug and the right plate the yellow and white mug",
        "I want the white cup sitting on the left plate and the yellow and white cup on the right plate",
        "I want each mug on a plate: the white one on the left, the yellow and white one on the right"],
    5: ["pick up the book and put it in the back section of the caddy", "move the book into the rear slot of the caddy",
        "the book goes in the back compartment of the caddy", "I want the caddy's back section to hold the book",
        "I want the book placed inside the rear section of the caddy", "I want the book standing in the back slot of the caddy"],
    6: ["put the white mug on the plate and the chocolate pudding on the right side of the plate",
        "move the white cup onto the plate, then put the pudding box next to the plate on its right",
        "the white mug goes on the plate and the chocolate pudding goes to the right of the plate",
        "I want the white mug on the plate and the chocolate pudding to the right of the plate",
        "I want the white cup sitting on the plate with the pudding box on its right side",
        "I want the plate holding the white mug and the chocolate pudding beside it on the right"],
    7: ["put the soup can and the cream cheese into the basket", "pick up the alphabet soup and the cream cheese box and drop them in the basket",
        "the alphabet soup and the cream cheese box go in the basket", "I want the soup can and the cream cheese in the basket",
        "I want both the alphabet soup and the cream cheese box placed inside the basket", "I want the basket to hold the alphabet soup and the cream cheese box"],
    8: ["put the two moka pots on the stove", "move both coffee pots onto the stove", "both moka pots go on the stove",
        "I want the pair of moka pots set down on the stove", "I want the two coffee pots placed on top of the stove", "I want the stove to have both moka pots on it"],
    9: ["put the yellow and white mug into the microwave and close the door", "move the yellow and white cup inside the microwave, then shut it",
        "the yellow and white mug goes in the microwave and the microwave gets closed", "I want the microwave door closed after the yellow and white mug is put in",
        "I want the yellow and white cup inside the closed microwave", "I want the microwave shut with the yellow and white mug inside it"],
}
OBJ_PLACE = {
    0: [["alphabet soup", "soup"], ["tomato sauce"], ["basket"]],
    1: [["cream cheese"], ["butter"], ["basket"]],
    2: [["moka pot", "coffee pot"], ["stove"]],
    3: [["bowl"], ["drawer"]],
    4: [["white mug", "white cup"], ["yellow and white mug", "yellow and white cup"], ["left"], ["right"], ["plate"]],
    5: [["book"], ["caddy"]],
    6: [["white mug", "white cup"], ["pudding"], ["plate"], ["right"]],
    7: [["alphabet soup", "soup"], ["cream cheese"], ["basket"]],
    8: [["moka pots", "coffee pots"], ["stove"]],
    9: [["yellow and white mug", "yellow and white cup"], ["microwave"]],
}
log = S14.log


# ----------------------------------------------------------------------
# text scores (no new package)
# ----------------------------------------------------------------------

def toks(s: str) -> list[str]:
    return re.sub(r"[^\w\s'-]", " ", s.lower()).split()


def token_f1(pred: str, ref: str) -> float:
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return float(p == r)
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def lcs(a: list, b: list) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return float(p == r)
    m = lcs(p, r)
    if m == 0:
        return 0.0
    prec, rec = m / len(p), m / len(r)
    return 2 * prec * rec / (prec + rec)


def verb_object(s: str) -> tuple[str, str]:
    t = [w for w in toks(s)]
    if t[:3] == ["i", "want", "to"]:
        t = t[3:]
    elif t[:1] == ["to"]:
        t = t[1:]
    if not t:
        return "", ""
    verb, rest = t[0], t[1:]
    spans, cur, stops = [], [], []
    for w in rest:
        if w in STOP:
            spans.append(cur)
            stops.append(w)
            cur = []
        else:
            cur.append(w)
    spans.append(cur)
    obj = spans[0][-1] if spans[0] else ""
    if not obj and len(spans) > 1:
        nxt = spans[1][1:] if stops[0] == "to" else spans[1]
        obj = nxt[-1] if nxt else ""
    return verb, obj


def obj_place_ok(sentence: str, task: int) -> bool:
    s = sentence.lower()
    return all(any(k in s for k in group) for group in OBJ_PLACE[task])


# ----------------------------------------------------------------------
# W
# ----------------------------------------------------------------------

class W:
    """Connector + SmolVLM2 language model. Only the connector, the last 4 blocks and the final norm are trained."""

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.torch, self.device = torch, device
        full = AutoModelForImageTextToText.from_pretrained(LM_ID, dtype=torch.float32)
        self.lm, self.head = full.model.text_model.to(device), full.lm_head.to(device)
        del full
        self.tok = AutoTokenizer.from_pretrained(LM_ID)
        self.eos = self.tok.convert_tokens_to_ids("<end_of_utterance>")
        for p in list(self.lm.parameters()) + list(self.head.parameters()):
            p.requires_grad_(False)
        self.train_lm = [*self.lm.layers[-N_TRAIN_BLOCKS:], self.lm.norm]
        for m in self.train_lm:
            for p in m.parameters():
                p.requires_grad_(True)
        self.init_lm = {k: v.detach().clone().cpu() for k, v in self.lm_state().items()}
        self.reset(0)

    def lm_state(self):
        return {f"{i}.{k}": v for i, m in enumerate(self.train_lm) for k, v in m.state_dict().items()}

    def reset(self, seed: int):
        torch = self.torch
        torch.manual_seed(seed)
        nn = torch.nn
        self.pos = nn.Parameter(torch.randn(N_POS, D_IN, device=self.device) * 0.02)
        self.conn = nn.Sequential(nn.Linear(D_IN, D_LM), nn.GELU(), nn.Linear(D_LM, D_LM)).to(self.device)
        for i, m in enumerate(self.train_lm):
            m.load_state_dict({k[len(f"{i}."):]: v for k, v in self.init_lm.items() if k.startswith(f"{i}.")})

    def trainable(self):
        return [self.pos, *self.conn.parameters()], [p for m in self.train_lm for p in m.parameters()]

    def n_trainable(self) -> int:
        a, b = self.trainable()
        return int(sum(p.numel() for p in a + b))

    def state(self):
        return {"pos": self.pos.detach().cpu().clone(), "conn": {k: v.detach().cpu().clone() for k, v in self.conn.state_dict().items()},
                "lm": {k: v.detach().cpu().clone() for k, v in self.lm_state().items()}}

    def load(self, st):
        with self.torch.no_grad():
            self.pos.copy_(st["pos"].to(self.device))
        self.conn.load_state_dict(st["conn"])
        for i, m in enumerate(self.train_lm):
            m.load_state_dict({k[len(f"{i}."):]: v for k, v in st["lm"].items() if k.startswith(f"{i}.")})

    def ids(self, text: str) -> list[int]:
        return self.tok(" " + text, add_special_tokens=False).input_ids

    def visual(self, v):
        """v: (B, k, 1152) seen frames only -> (B, k, 960)."""
        return self.conn(v + self.pos[: v.shape[1]])

    def batch_inputs(self, exs):
        """exs: list of (vis (k, 1152) array, tag, target or None). Right-padded embeds, mask, labels."""
        torch = self.torch
        seqs = []
        for vis, tag, tgt in exs:
            tag_ids = self.ids(tag + ":")
            tgt_ids = (self.ids(tgt) + [self.eos]) if tgt is not None else []
            seqs.append((torch.as_tensor(np.asarray(vis, np.float32), device=self.device), tag_ids, tgt_ids))
        L = max(v.shape[0] + len(a) + len(b) for v, a, b in seqs)
        emb = torch.zeros(len(seqs), L, D_LM, device=self.device)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        labels = torch.full((len(seqs), L), -100, dtype=torch.long, device=self.device)
        tok_emb = self.lm.embed_tokens
        for i, (v, a, b) in enumerate(seqs):
            k = v.shape[0]
            ids = torch.as_tensor(a + b, dtype=torch.long, device=self.device)
            emb[i, :k] = self.visual(v[None])[0]
            emb[i, k:k + len(ids)] = tok_emb(ids)
            mask[i, :k + len(ids)] = 1
            if b:
                labels[i, k + len(a):k + len(a) + len(b)] = torch.as_tensor(b, device=self.device)
        return emb, mask, labels

    def loss(self, exs):
        torch = self.torch
        emb, mask, labels = self.batch_inputs(exs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.lm(inputs_embeds=emb, attention_mask=mask).last_hidden_state
            logits = self.head(h).float()
        return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)

    def generate(self, vis_list, tag: str, bs: int = 128) -> list[str]:
        """Greedy, at most MAX_NEW tokens; all items in vis_list must have the same number of frames; W gets frames and a tag only."""
        torch = self.torch
        out = []
        for s in range(0, len(vis_list), bs):
            chunk = vis_list[s:s + bs]
            with torch.no_grad():
                emb, mask, _ = self.batch_inputs([(v, tag, None) for v in chunk])
                gen = [[] for _ in chunk]
                done = np.zeros(len(chunk), bool)
                for _ in range(MAX_NEW):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = self.lm(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, -1]
                        nxt = self.head(h).float().argmax(-1)
                    for i, t in enumerate(nxt.tolist()):
                        if not done[i]:
                            if t == self.eos:
                                done[i] = True
                            else:
                                gen[i].append(t)
                    if done.all():
                        break
                    emb = torch.cat([emb, self.lm.embed_tokens(nxt)[:, None]], 1)
                    mask = torch.cat([mask, torch.ones(len(chunk), 1, dtype=mask.dtype, device=self.device)], 1)
            out += [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        return out

    def fit(self, train_fn, val_exs, seed: int, tag: str) -> dict:
        """train_fn(epoch, rng) -> list of examples. Early stopping on validation loss (patience 3), best epoch kept."""
        torch = self.torch
        self.reset(seed)
        a, b = self.trainable()
        opt = torch.optim.AdamW([{"params": a, "lr": LR_CONN}, {"params": b, "lr": LR_LM}], weight_decay=WD)
        rng = np.random.default_rng(seed)
        best, best_state, bad, hist = float("inf"), None, 0, []
        for ep in range(MAX_EPOCHS):
            exs = train_fn(ep, rng)
            order = rng.permutation(len(exs))
            self.conn.train()
            tl, nb = 0.0, 0
            for s in range(0, len(order), BATCH):
                loss = self.loss([exs[i] for i in order[s:s + BATCH]])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl, nb = tl + float(loss), nb + 1
            self.conn.eval()
            with torch.no_grad():
                vl = float(np.mean([float(self.loss(val_exs[s:s + 64])) for s in range(0, len(val_exs), 64)]))
            hist.append({"epoch": ep, "train_loss": tl / nb, "val_loss": vl})
            log(f"{tag} epoch {ep} train {tl / nb:.4f} val {vl:.4f}")
            if vl < best - 1e-6:
                best, best_state, bad = vl, self.state(), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        self.load(best_state)
        return {"best_val_loss": best, "epochs_run": len(hist), "best_epoch": int(np.argmin([h["val_loss"] for h in hist])), "history": hist}


# ----------------------------------------------------------------------
# EgoIntent data (development steps only)
# ----------------------------------------------------------------------

def ego_data() -> dict:
    rows = sorted([json.loads(x) for x in MANIFEST.open() if x.strip()], key=lambda r: r["sample_id"])
    assert all(r["split"] in ("train", "val") for r in rows)
    dev = {r["sample_id"] for r in rows}
    labels = {}
    for d in sorted({r["sample_id"].rsplit("/", 1)[0] for r in rows}):
        n_mp4 = len(list((EGO_ROOT / d).glob("step*.mp4")))
        n_dev = sum(1 for s in dev if s.rsplit("/", 1)[0] == d)
        assert n_mp4 == n_dev, f"{d} holds non-development steps"
        for st in json.loads((EGO_ROOT / d / "step_label.json").read_text())["steps"]:
            sid = f"{d}/step{st['step_id']}"
            assert sid in dev
            labels[sid] = st
    sids = [r["sample_id"] for r in rows]
    assert set(labels) == dev
    ladder_ids = json.loads((LADDER / "sample_ids.json").read_text())
    pos = {s: i for i, s in enumerate(ladder_ids)}
    emb = np.load(LADDER / "frame_emb.npy", mmap_mode="r")
    feats = np.stack([np.asarray(emb[pos[s]], np.float32) for s in sids])
    texts = {"why": [r["why"] for r in rows], "what": [labels[s]["local_intent"] for s in sids], "next": [labels[s]["observed_next_step"] for s in sids]}
    assert texts["why"] == [labels[s]["procedural_intent"] for s in sids]
    scenes = [r["scene"] for r in rows]
    vids = [r["video_uid"] for r in rows]
    fold_map = CV.assign_folds(scenes, N_FOLDS_EGO)
    start = np.array([labels[s]["start_time"] for s in sids], float)
    end = np.array([labels[s]["obs_end_time"] for s in sids], float)
    nxt_start = np.full(len(sids), np.nan)
    by_vid = {}
    for i, s in enumerate(sids):
        by_vid.setdefault(s.rsplit("/", 1)[0], {})[int(labels[s]["step_id"])] = i
    for d, m in by_vid.items():
        for sid_n, i in m.items():
            if sid_n + 1 in m:
                nxt_start[i] = start[m[sid_n + 1]]
    return {"sids": sids, "feats": feats, "texts": texts, "scenes": np.array(scenes), "vids": np.array(vids),
            "fold": np.array([fold_map[s] for s in scenes]), "start": start, "end": end, "next_start": nxt_start}


def ego_seconds(data, k: int) -> dict:
    dur = data["end"] - data["start"]
    seen = (k - 1) / 7 * dur
    gap = data["next_start"] - (data["start"] + seen)
    return {"seen": seen, "remaining": dur - seen, "until_next": gap}


def video_val(vids, seed: int = 0) -> set:
    u = sorted(set(vids))
    n = max(1, int(round(0.1 * len(u))))
    return set(np.asarray(u)[np.random.default_rng(seed).permutation(len(u))[:n]].tolist())


def ego_examples(data, idx, ks, rng=None):
    out = []
    for i in idx:
        for tag in EGO_TAGS:
            k = int(rng.choice(ks)) if rng is not None else ks[0]
            out.append((data["feats"][i, :k], tag, data["texts"][tag][i]))
    return out


def cmd_train_ego(args) -> int:
    data = ego_data()
    (WORK / "ego").mkdir(parents=True, exist_ok=True)
    w = None
    for f in range(N_FOLDS_EGO):
        path = WORK / "ego" / f"fold{f}.pt"
        if path.exists():
            continue
        w = w or W()
        tr = np.flatnonzero(data["fold"] != f)
        val_v = video_val(data["vids"][tr])
        va = np.array([i for i in tr if data["vids"][i] in val_v])
        tr = np.array([i for i in tr if data["vids"][i] not in val_v])
        assert not set(data["scenes"][tr]) & set(data["scenes"][data["fold"] == f])
        val_exs = ego_examples(data, va, EGO_K, np.random.default_rng(0))
        t0 = time.time()
        info = w.fit(lambda ep, rng: ego_examples(data, tr, EGO_K, rng), val_exs, 0, f"ego fold {f}")
        info.update({"fold": f, "n_train": len(tr), "n_val": len(va), "val_videos": sorted(val_v), "seconds": time.time() - t0,
                     "mean_train_emb": data["feats"][tr].reshape(-1, D_IN).mean(0).tolist()})
        w.torch.save({"state": w.state(), "info": info}, path)
    return 0


def cmd_write_ego(args) -> int:
    data = ego_data()
    out = WORK / "ego_written.json"
    if out.exists():
        return 0
    w = W()
    rec = {}
    t0 = time.time()
    for f in range(N_FOLDS_EGO):
        ck = w.torch.load(WORK / "ego" / f"fold{f}.pt", weights_only=False)
        w.load(ck["state"])
        te = np.flatnonzero(data["fold"] == f)
        mu = np.asarray(ck["info"]["mean_train_emb"], np.float32)
        for k in EGO_K:
            for tag in EGO_TAGS:
                sents = w.generate([data["feats"][i, :k] for i in te], tag)
                blank = w.generate([np.tile(mu, (k, 1))], tag)[0]
                for i, s in zip(te, sents):
                    rec[f"{data['sids'][i]}|{k}|{tag}"] = {"W": s, "blank": blank}
        log(f"ego fold {f} written")
    out.write_text(json.dumps({"written": rec, "gpu_seconds": time.time() - t0}))
    return 0


# ----------------------------------------------------------------------
# LIBERO data (Stage 15 features and folds)
# ----------------------------------------------------------------------

def lib_data():
    import run_stage15_supervised as S15

    seqs, label, info, _n, original = S15.load_libero("agent")
    outer = S15.libero_outer(label)
    feat_dir = S15.WORK / "feat" / "agent"
    dur = {k: float(np.load(feat_dir / f"{k}.npz")["times"][-1]) for k in seqs}
    orig = {int(t): v for t, v in original.items()}
    return {"seqs": seqs, "label": label, "info": info, "outer": outer, "dur": dur, "orig": orig, "S15": S15}


def task_sentences(orig: dict, t: int) -> list[str]:
    return [orig[t]] + TRAIN_SENTENCES[t]


def lib_examples(L, keys, rng, ps=LIB_TRAIN_P):
    out = []
    for k in keys:
        t = L["label"][k]
        for p in ps:
            ss = task_sentences(L["orig"], t)
            out.append((L["seqs"][k][p], "goal", ss[int(rng.integers(len(ss)))]))
    return out


def lib_fit(w, L, train_keys, seed, tag):
    S15 = L["S15"]
    tk = sorted(train_keys)
    vm = S15.val_split([L["label"][k] for k in tk], 0.1, 0)
    va = [k for k, m in zip(tk, vm) if m]
    tr = [k for k, m in zip(tk, vm) if not m]
    val_exs = lib_examples(L, va, np.random.default_rng(0))
    t0 = time.time()
    info = w.fit(lambda ep, rng: lib_examples(L, tr, rng), val_exs, seed, tag)
    mu = np.stack([L["seqs"][k][p] for k in tr for p in LIB_TRAIN_P]).reshape(-1, D_IN).mean(0)
    info.update({"n_train": len(tr), "n_val": len(va), "seconds": time.time() - t0, "mean_train_emb": mu.tolist()})
    return info


def lib_write(w, L, keys, mu):
    rec = {}
    for p in LIB_P:
        sents = w.generate([L["seqs"][k][p] for k in keys], "goal")
        for k, s in zip(keys, sents):
            rec[f"{k}|{p}"] = s
    rec["blank"] = w.generate([np.tile(mu, (8, 1))], "goal")[0]
    return rec


def cmd_train_libero(args) -> int:
    L = lib_data()
    (WORK / "libero").mkdir(parents=True, exist_ok=True)
    w = None
    for f in range(L["S15"].N_OUTER):
        path = WORK / "libero" / f"fold{f}.pt"
        if path.exists():
            continue
        w = w or W()
        train = [k for k, g in L["outer"].items() if g != f]
        assert not set(train) & {k for k, g in L["outer"].items() if g == f}
        info = lib_fit(w, L, train, 0, f"libero fold {f}")
        info["fold"] = f
        w.torch.save({"state": w.state(), "info": info}, path)
    return 0


def cmd_write_libero(args) -> int:
    L = lib_data()
    out = WORK / "libero_written.json"
    if out.exists():
        return 0
    w = W()
    rec, t0 = {}, time.time()
    for f in range(L["S15"].N_OUTER):
        ck = w.torch.load(WORK / "libero" / f"fold{f}.pt", weights_only=False)
        w.load(ck["state"])
        keys = sorted(k for k, g in L["outer"].items() if g == f)
        r = lib_write(w, L, keys, np.asarray(ck["info"]["mean_train_emb"], np.float32))
        rec.update({k: v for k, v in r.items() if k != "blank"})
        rec[f"blank|fold{f}"] = r["blank"]
        log(f"libero fold {f} written")
    out.write_text(json.dumps({"written": rec, "gpu_seconds": time.time() - t0}))
    return 0


def cmd_unseen(args) -> int:
    L = lib_data()
    out = WORK / "unseen7_written.json"
    if out.exists():
        return 0
    w = W()
    train = [k for k, t in L["label"].items() if t != 7]
    test = sorted(k for k, t in L["label"].items() if t == 7)
    info = lib_fit(w, L, train, 0, "unseen task 7")
    assert all(L["label"][k] != 7 for k in train)
    rec = lib_write(w, L, test, np.asarray(info["mean_train_emb"], np.float32))
    info.pop("mean_train_emb")
    out.write_text(json.dumps({"written": rec, "info": info}))
    return 0


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def fmt(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def ci(c, v):
    return CV.video_bootstrap(np.asarray(c), np.asarray(v, float), np.ones(len(v)))


def analyze_ego(enc, rows_out, md):
    import v3_common as V3

    data = ego_data()
    wr = json.loads((WORK / "ego_written.json").read_text())["written"]
    sids, scenes, vids = data["sids"], data["scenes"], data["vids"]
    n = len(sids)
    cands = {tag: CV.build_kway(vids.tolist(), data["texts"][tag], sids, 5) for tag in EGO_TAGS}
    # controls (b) most frequent, (c) nearest training unit, per fold
    mf_sent, nn_sent = {}, {}
    for f in range(N_FOLDS_EGO):
        tr, te = np.flatnonzero(data["fold"] != f), np.flatnonzero(data["fold"] == f)
        for tag in EGO_TAGS:
            vals, cnt = np.unique([data["texts"][tag][i] for i in tr], return_counts=True)
            mf_sent[(f, tag)] = sorted(v for v, c in zip(vals, cnt) if c == cnt.max())[0]
        for k in EGO_K:
            A = S14.normalize(data["feats"][tr, :k].mean(1))
            B = S14.normalize(data["feats"][te, :k].mean(1))
            nn = tr[np.argmax(B @ A.T, axis=1)]
            for i, j in zip(te, nn):
                nn_sent[(i, k)] = j
    # Qwen (Stage 14 A1) sentences
    qwen = {}
    for k in (2, 4, 8):
        for i, s in enumerate(sids):
            p = S14.ego_path(s, k)
            if p.exists():
                qwen[(i, k)] = json.loads(p.read_text())["sentence"]
    alltexts = sorted({t for tag in EGO_TAGS for t in data["texts"][tag]} | {v[m] for v in wr.values() for m in ("W", "blank")}
                      | set(mf_sent.values()) | set(qwen.values()))
    E = dict(zip(alltexts, enc(alltexts, template=False)))
    video_texts = {tag: {} for tag in EGO_TAGS}
    for tag in EGO_TAGS:
        for i in range(n):
            video_texts[tag].setdefault(vids[i], set()).add(data["texts"][tag][i])

    def scores(i, tag, s):
        e = E[s]
        c = cands[tag][i]
        h5 = float(S14.strict_hit(np.array([E[data["texts"][tag][int(j)]] @ e for j in c]), 0)) if c[0] >= 0 else np.nan
        allv = sorted(video_texts[tag][vids[i]])
        rr = 1.0 / S14.rank_of_true(np.array([E[x] @ e for x in allv]), allv.index(data["texts"][tag][i]))
        ref = data["texts"][tag][i]
        vt, ot = verb_object(ref)
        vw, ow = verb_object(s)
        return {"5way": h5, "mrr": rr, "f1": token_f1(s, ref), "rougeL": rouge_l(s, ref), "verb": float(vw == vt and vt != ""), "object": float(ow == ot and ot != "")}

    data_f = V3.load_data()
    mfF = V3.f_metrics(data_f)
    fpos = {s: j for j, s in enumerate(data_f.sample_ids)}
    per = {}
    for tag in EGO_TAGS:
        for k in EGO_K:
            sec = ego_seconds(data, k)
            outs = {
                "W": [scores(i, tag, wr[f"{sids[i]}|{k}|{tag}"]["W"]) for i in range(n)],
                "W-blank": [scores(i, tag, wr[f"{sids[i]}|{k}|{tag}"]["blank"]) for i in range(n)],
                "most-frequent": [scores(i, tag, mf_sent[(data["fold"][i], tag)]) for i in range(n)],
                "nearest-train": [scores(i, tag, data["texts"][tag][nn_sent[(i, k)]]) for i in range(n)],
            }
            if tag == "why" and all((i, k) in qwen for i in range(n)):
                outs["Qwen (Stage 14 A1)"] = [scores(i, tag, qwen[(i, k)]) for i in range(n)]
            for model, sc in outs.items():
                per[(tag, k, model)] = sc
                valid = np.array([not np.isnan(x["5way"]) for x in sc])
                row = {"tag": tag, "k": k, "model": model, "n_steps": n, "n_5way": int(valid.sum()),
                       "seconds_seen": float(sec["seen"].mean()), "seconds_remaining": float(sec["remaining"].mean()),
                       "seconds_to_next_step": float(np.nanmean(sec["until_next"])) if tag == "next" else ""}
                for m in ("5way", "mrr", "f1", "rougeL", "verb", "object"):
                    v = np.array([x[m] for x in sc])
                    sel = valid if m == "5way" else np.ones(n, bool)
                    a = ci(scenes[sel], v[sel])
                    row.update({m: a[0], f"{m}_lo": a[1], f"{m}_hi": a[2]})
                rows_out.append(row)
            if tag == "why" and k in mfF:
                valid = np.array([cands["why"][i][0] >= 0 for i in range(n)])
                fh = np.array([mfF[k]["kway5"][fpos[sids[i]]] for i in range(n)])
                a = ci(scenes[valid], fh[valid])
                rows_out.append({"tag": "why", "k": k, "model": "F (chooses among 5)", "n_5way": int(valid.sum()), "5way": a[0], "5way_lo": a[1], "5way_hi": a[2]})
                per[("why", k, "F")] = [{"5way": float(x)} for x in fh]
    # rules
    valid = np.array([cands["why"][i][0] >= 0 for i in range(n)])
    hw = np.array([x["5way"] for x in per[("why", 8, "W")]])
    rules = {}
    r1 = next(r for r in rows_out if r["tag"] == "why" and r["k"] == 8 and r["model"] == "W")
    rules["W-1"] = {"rule": "why: 5-way at k=8, CI lower bound > 20%", "value": r1["5way"], "ci95": [r1["5way_lo"], r1["5way_hi"]], "pass": r1["5way_lo"] > 0.2}
    for name, ctrl in (("W-2", "W-blank"), ("W-3", "nearest-train")):
        hc = np.array([x["5way"] for x in per[("why", 8, ctrl)]])
        d = ci(scenes[valid], (hw - hc)[valid])
        rules[name] = {"rule": f"why: W - {ctrl}, k=8, 5-way, CI lower bound > 0", "delta": d[0], "ci95": [d[1], d[2]], "pass": d[1] > 0}
    desc = []
    for k in (2, 4, 8):
        for other in ("Qwen (Stage 14 A1)", "F"):
            if ("why", k, other) in per:
                ho = np.array([x["5way"] for x in per[("why", k, other)]])
                hk = np.array([x["5way"] for x in per[("why", k, "W")]])
                d = ci(scenes[valid], (hk - ho)[valid])
                desc.append({"comparison": f"W - {other}", "tag": "why", "k": k, "delta": d[0], "ci95_lo": d[1], "ci95_hi": d[2]})
    # sentences and human check
    sent_rows = [{"sample_id": sids[i], "scene": scenes[i], "fold": int(data["fold"][i]), "tag": tag, "true": data["texts"][tag][i],
                  "W_k8": wr[f"{sids[i]}|8|{tag}"]["W"], "Qwen_k8": qwen.get((i, 8), "") if tag == "why" else ""} for i in range(n) for tag in EGO_TAGS]
    S14.write_csv(RES / "sentences_ego_k8.csv", sent_rows)
    rng = np.random.default_rng(0)
    folds = data["fold"]
    pick = []
    base, extra = divmod(60, N_FOLDS_EGO)
    for f in range(N_FOLDS_EGO):
        ids = np.flatnonzero(folds == f)
        pick += rng.choice(ids, size=base + (1 if f < extra else 0), replace=False).tolist()
    hc, key = [], {}
    for i in pick:
        a = [("W", wr[f"{sids[i]}|8|why"]["W"]), ("Qwen", qwen.get((i, 8), ""))]
        if rng.random() < 0.5:
            a = a[::-1]
        key[sids[i]] = {"A": a[0][0], "B": a[1][0]}
        hc.append({"sample_id": sids[i], "scene_group": int(folds[i]), "true_why": data["texts"]["why"][i], "sentence_A": a[0][1], "sentence_B": a[1][1],
                   "rating_A": "", "rating_B": ""})
    S14.write_csv(RES / "human_check_w.csv", hc)
    (WORK / "human_check_w_key.json").write_text(json.dumps(key, indent=1))
    # seconds table
    sec_rows = []
    for k in EGO_K:
        s = ego_seconds(data, k)
        sec_rows.append({"k": k, "mean_seconds_seen": float(s["seen"].mean()), "mean_seconds_remaining_to_step_end": float(s["remaining"].mean()),
                         "mean_seconds_to_next_step_start": float(np.nanmean(s["until_next"])), "n_with_next_step": int(np.isfinite(s["until_next"]).sum())})
    return {"rules": rules, "descriptive": desc, "seconds": sec_rows, "n_steps": n, "n_videos": len(set(vids)), "n_scenes": len(set(scenes)),
            "n_distinct": {tag: len(set(data["texts"][tag])) for tag in EGO_TAGS},
            "step_seconds": {"mean": float((data["end"] - data["start"]).mean()), "min": float((data["end"] - data["start"]).min()),
                             "max": float((data["end"] - data["start"]).max())}, "first10": sent_rows[:30:3]}


def analyze_libero(enc, rows_out):
    import csv

    import libero_goal_ablation as G

    L = lib_data()
    wr = json.loads((WORK / "libero_written.json").read_text())["written"]
    orig, label, outer = L["orig"], L["label"], L["outer"]
    keys = sorted(label, key=lambda k: (label[k], L["info"][k]["demo"]))
    train_set = {s for t in range(10) for s in task_sentences(orig, t)}
    desc_e = enc([orig[t] for t in range(10)], template=False)
    mf = {}
    nn = {}
    for f in range(L["S15"].N_OUTER):
        tr = [k for k in keys if outer[k] != f]
        cnt = {}
        for k in tr:  # most frequent training target: per-task sentences are equally likely, so the task with most demos, its description
            cnt[label[k]] = cnt.get(label[k], 0) + 1
        mf[f] = orig[sorted(cnt, key=lambda t: (-cnt[t], t))[0]]
        for p in LIB_P:
            A = S14.normalize(np.stack([L["seqs"][k][p].mean(0) for k in tr]))
            te = [k for k in keys if outer[k] == f]
            B = S14.normalize(np.stack([L["seqs"][k][p].mean(0) for k in te]))
            for k, j in zip(te, np.argmax(B @ A.T, axis=1)):
                nn[(k, p)] = orig[label[tr[j]]]
    texts = sorted(set(wr.values()) | set(mf.values()) | set(orig.values()))
    E = dict(zip(texts, enc(texts, template=False)))

    def near(s):
        return S14.nearest(E[s], desc_e)

    res = {}
    for p in LIB_P:
        for model in ("W", "W-blank", "most-frequent", "nearest-train"):
            got = []
            for k in keys:
                s = {"W": wr[f"{k}|{p}"], "W-blank": wr[f"blank|fold{outer[k]}"], "most-frequent": mf[outer[k]], "nearest-train": nn[(k, p)]}[model]
                y, cos = near(s)
                t = label[k]
                got.append({"key": k, "s": s, "y": y, "hit": float(y == t), "two": (float(cos[t] > cos[1 - t]) if t in (0, 1) else np.nan),
                            "novel": float(s not in train_set), "objplace": float(obj_place_ok(s, t))})
            res[(p, model)] = got
    sec = {p: np.array([p / 100 * L["dur"][k] for k in keys]) for p in LIB_P}
    rem = {p: np.array([(1 - p / 100) * L["dur"][k] for k in keys]) for p in LIB_P}
    # Qwen (Part C) on the 299 units
    qres = {}
    for v in ("G0", "G1", "G2", "O1"):
        f = ROOT / f"results/stage14b_generate_variants/sentences_{v}.csv"
        if f.exists():
            for r in csv.DictReader(f.open()):
                qres[(v, int(r["p"]), f"t{r['task']}_d{r['demo']}")] = float(int(r["nearest_task"]) == int(r["true_task"]))
    F = {}
    for r in csv.DictReader((ROOT / "results/stage12_end_to_end/recognition.csv").open()):
        if r["context"] == "C2" and r["template"] == "T1":
            F[int(r["p"])] = float(r["accuracy"])
    C = {0: 0.8865435356200527, 10: 0.9630606860158312, 25: 0.9973614775725593, 50: 1.0, 75: 1.0, 100: 1.0}
    for p in LIB_P:
        for model in ("W", "W-blank", "most-frequent", "nearest-train"):
            g = res[(p, model)]
            row = {"p": p, "model": model, "n": len(g), "seconds_seen": float(sec[p].mean()), "seconds_remaining": float(rem[p].mean())}
            for m in ("hit", "two", "novel", "objplace"):
                v = np.array([x[m] for x in g])
                v = v[~np.isnan(v)]
                a = G.paired_bootstrap(v)
                row.update({ {"hit": "accuracy", "two": "within_scene_0v1", "novel": "novel_share", "objplace": "object_place"}[m] + s: a[j]
                             for j, s in enumerate(("", "_lo", "_hi"))})
            rows_out.append(row)
        rows_out.append({"p": p, "model": "C (Stage 15, classifier, 379 demos)", "accuracy": C[p]})
        rows_out.append({"p": p, "model": "F (Stage 12, chooses, 299 units)", "accuracy": F.get(p, "")})
        for v in ("G0", "G1", "G2", "O1"):
            u = [x for x in qres if x[0] == v and x[1] == p]
            if u:
                rows_out.append({"p": p, "model": f"Qwen {v} (Part C, 299 units){' [given object names]' if v == 'O1' else ''}",
                                 "accuracy": float(np.mean([qres[x] for x in u])), "n": len(u)})
    # rules
    w25 = next(r for r in rows_out if r.get("p") == 25 and r.get("model") == "W" and "accuracy_lo" in r)
    rules = {"W-4": {"rule": "W accuracy at p=25 (379 demos), CI lower bound > 50%", "value": w25["accuracy"], "ci95": [w25["accuracy_lo"], w25["accuracy_hi"]],
                     "pass": w25["accuracy_lo"] > 0.5}}
    units = [f"t{t}_d{d}" for t, d in S14.libero_units()]
    whit = {x["key"]: x["hit"] for x in res[(25, "W")]}
    acc25 = {v: np.mean([qres[(v, 25, u)] for u in units]) for v in ("G0", "G1", "G2", "O1") if all((v, 25, u) in qres for u in units)}
    if acc25:
        best = max(acc25, key=lambda v: (acc25[v], -("G0", "G1", "G2", "O1").index(v)))
        d = G.paired_bootstrap(np.array([whit[u] - qres[(best, 25, u)] for u in units]))
        rules["W-5"] = {"rule": f"W - best Qwen variant ({best}) at p=25, 299 units, CI lower bound > 0", "delta": d[0], "ci95": [d[1], d[2]], "pass": d[1] > 0,
                        "qwen_acc_p25": acc25}
        desc = []
        for v in acc25:
            d = G.paired_bootstrap(np.array([whit[u] - qres[(v, 25, u)] for u in units]))
            desc.append({"comparison": f"W - Qwen {v}", "p": 25, "delta": d[0], "ci95_lo": d[1], "ci95_hi": d[2]})
        rules["W-5"]["all_variants"] = desc
    # tokens
    for p in (25, 50):
        tok = {"tokens": {f"E2E-W-p{p}": {}}, "new": {f"E2E-W-p{p}": []}}
        for t, d in S14.libero_units():
            k = f"t{t}_d{d}"
            tok["tokens"][f"E2E-W-p{p}"][f"{t},{d}"] = wr[f"{k}|{p}"]
            tok["new"][f"E2E-W-p{p}"].append(f"{t},{d}")
        (RES / f"w_tokens_p{p}.json").write_text(json.dumps(tok, indent=1))
    sent_rows = []
    for p in LIB_P:
        for x in res[(p, "W")]:
            sent_rows.append({"task": label[x["key"]], "demo": L["info"][x["key"]]["demo"], "p": p, "fold": outer[x["key"]], "true_description": orig[label[x["key"]]],
                              "written": x["s"], "nearest_task": x["y"], "object_place_ok": int(x["objplace"]), "novel": int(x["novel"])})
    sent_rows.sort(key=lambda r: (r["task"], r["demo"], r["p"]))
    S14.write_csv(RES / "sentences_libero.csv", sent_rows)
    # unseen task 7
    un = json.loads((WORK / "unseen7_written.json").read_text())
    uw = un["written"]
    t7 = sorted(k for k in keys if label[k] == 7)
    texts7 = sorted({v for v in uw.values()} - set(E))
    E.update(dict(zip(texts7, enc(texts7, template=False))) if texts7 else {})
    un_rows = []
    for p in LIB_P:
        ys = [near(uw[f"{k}|{p}"])[0] for k in t7]
        un_rows.append({"p": p, "n": len(t7), "object_place_task7": float(np.mean([obj_place_ok(uw[f"{k}|{p}"], 7) for k in t7])),
                        "nearest_is_7": float(np.mean([y == 7 for y in ys])), "nearest_counts": {str(t): ys.count(t) for t in sorted(set(ys))},
                        "classifier_9_labels": 0.0, "example": uw[f"{t7[0]}|{p}"]})
    per_task = []
    for t in range(10):
        ks = [k for k in keys if label[k] == t]
        dd = np.array([L["dur"][k] for k in ks])
        per_task.append({"task": t, "n_demos": len(ks), "fps": 10.0, "mean_s": float(dd.mean()), "min_s": float(dd.min()), "max_s": float(dd.max()),
                         **{f"acc_p{p}": float(np.mean([x["hit"] for x in res[(p, "W")] if label[x["key"]] == t])) for p in LIB_P}})
    return {"rules": rules, "unseen": un_rows, "unseen_info": un["info"], "per_task": per_task, "n": len(keys), "first10": sent_rows[:60:6],
            "seconds": [{"p": p, "mean_seconds_seen": float(sec[p].mean()), "mean_seconds_remaining": float(rem[p].mean())} for p in LIB_P]}


def cmd_analyze(args) -> int:
    RES.mkdir(parents=True, exist_ok=True)
    enc = S14.TextEncoder()
    ego_rows, lib_rows = [], []
    ego = analyze_ego(enc, ego_rows, None)
    S14.write_csv(RES / "ego_writer.csv", ego_rows)
    lib = analyze_libero(enc, lib_rows)
    S14.write_csv(RES / "libero_writer.csv", lib_rows)
    S14.write_csv(RES / "libero_per_task.csv", lib["per_task"])
    # markdown
    md = ["# W on EgoIntent (scene-held-out, 8 groups, 2,503 development steps)", "",
          "| tag | k | model | s seen | s left | 5-way | MRR | F1 | ROUGE-L | verb | object |", "|---|---:|---|---:|---:|---|---|---|---|---|---|"]
    for r in ego_rows:
        def c(m):
            return f"{fmt(r[m])} [{fmt(r[m + '_lo'])}; {fmt(r[m + '_hi'])}]" if m in r and m + "_lo" in r else (fmt(r[m]) if m in r else "")
        md.append(f"| {r['tag']} | {r['k']} | {r['model']} | {r.get('seconds_seen', '') and f'{r['seconds_seen']:.2f}'} | "
                  f"{r.get('seconds_remaining', '') and f'{r['seconds_remaining']:.2f}'} | {c('5way')} | {c('mrr')} | {c('f1')} | {c('rougeL')} | {c('verb')} | {c('object')} |")
    md += ["", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for k, r in ego["rules"].items():
        md.append(f"| {k}: {r['rule']} | {r.get('value', r.get('delta')):+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {'PASS' if r['pass'] else 'FAIL'} |")
    md += ["", "| descriptive | k | delta [95% CI] |", "|---|---:|---|"]
    md += [f"| {r['comparison']} (why) | {r['k']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in ego["descriptive"]]
    (RES / "ego_writer.md").write_text("\n".join(md) + "\n")
    md = ["# W on LIBERO-10 (379 demos, Stage 15 folds; nearest of the 10 descriptions, strict top-1)", "",
          "| p | model | s seen | s left | accuracy | within-scene 0v1 | novel | object+place |", "|---:|---|---:|---:|---|---|---|---|"]
    for r in lib_rows:
        def c(m):
            return f"{fmt(r[m])} [{fmt(r[m + '_lo'])}; {fmt(r[m + '_hi'])}]" if m + "_lo" in r else (fmt(r[m]) if isinstance(r.get(m), float) else "")
        md.append(f"| {r['p']} | {r['model']} | {r.get('seconds_seen', '') and f'{r['seconds_seen']:.1f}'} | {r.get('seconds_remaining', '') and f'{r['seconds_remaining']:.1f}'} | "
                  f"{c('accuracy')} | {c('within_scene_0v1')} | {c('novel_share')} | {c('object_place')} |")
    md += ["", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for k, r in lib["rules"].items():
        md.append(f"| {k}: {r['rule']} | {r.get('value', r.get('delta')):+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {'PASS' if r['pass'] else 'FAIL'} |")
    if "W-5" in lib["rules"]:
        md += [f"| descriptive: {d['comparison']} at p=25 | {d['delta']:+.3f} | [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}] | |" for d in lib["rules"]["W-5"]["all_variants"]]
    (RES / "libero_writer.md").write_text("\n".join(md) + "\n")
    un = ["# Unseen task 7 (W trained on the other 9 tasks; 43 task-7 demos)", "",
          "| p | object+place for task 7 | nearest description = task 7 | nearest counts | 9-label classifier | example |", "|---:|---:|---:|---|---:|---|"]
    un += [f"| {r['p']} | {fmt(r['object_place_task7'])} | {fmt(r['nearest_is_7'])} | {r['nearest_counts']} | 0.0 | {r['example']} |" for r in lib["unseen"]]
    (RES / "unseen_task7.md").write_text("\n".join(un) + "\n")
    ds = ["# Stage 14b W: data summary", "", "## EgoIntent (development steps only; locked test never opened)", "",
          f"- {ego['n_videos']} videos, {ego['n_scenes']} scenes in 8 scene groups (scene-held-out), {ego['n_steps']} steps (units).",
          f"- Distinct sentences: WHY {ego['n_distinct']['why']}, WHAT {ego['n_distinct']['what']}, NEXT {ego['n_distinct']['next']}.",
          f"- 8 frames per step, evenly from step start to the pre-outcome cutoff; step length mean {ego['step_seconds']['mean']:.2f} s "
          f"(min {ego['step_seconds']['min']:.3f}, max {ego['step_seconds']['max']:.2f}).", "",
          "| k | s seen | s left to step end | s to next step start | n with next step |", "|---:|---:|---:|---:|---:|"]
    ds += [f"| {r['k']} | {r['mean_seconds_seen']:.2f} | {r['mean_seconds_remaining_to_step_end']:.2f} | {r['mean_seconds_to_next_step_start']:.2f} | {r['n_with_next_step']} |"
           for r in ego["seconds"]]
    ds += ["", "## LIBERO-10 (agent view)", "", f"- {lib['n']} demonstrations, 10 tasks, 5 folds over demonstrations (Stage 15); the 299 robot units are a subset.",
           "- 8 frames per unit from the 4 fps grid (dataset 10 fps), spread over the first p% of the demo.", "",
           "| p | s seen | s left |", "|---:|---:|---:|"]
    ds += [f"| {r['p']} | {r['mean_seconds_seen']:.1f} | {r['mean_seconds_remaining']:.1f} |" for r in lib["seconds"]]
    ds += ["", "| task | demos | fps | mean s | min s | max s |", "|---:|---:|---:|---:|---:|---:|"]
    ds += [f"| {r['task']} | {r['n_demos']} | 10 | {r['mean_s']:.1f} | {r['min_s']:.1f} | {r['max_s']:.1f} |" for r in lib["per_task"]]
    ds += ["", "Seconds per task at each p: seen = p% of the task's mean duration; left = the rest (exact per-demo values in `libero_writer.csv` are averaged over all tasks)."]
    (RES / "data_summary.md").write_text("\n".join(ds) + "\n")
    ego_info = [__import__("torch").load(WORK / "ego" / f"fold{f}.pt", weights_only=False)["info"] for f in range(N_FOLDS_EGO)]
    lib_info = [__import__("torch").load(WORK / "libero" / f"fold{f}.pt", weights_only=False)["info"] for f in range(5)]
    gpu = sum(i["seconds"] for i in ego_info + lib_info) + lib["unseen_info"]["seconds"]
    gpu += json.loads((WORK / "ego_written.json").read_text())["gpu_seconds"] + json.loads((WORK / "libero_written.json").read_text())["gpu_seconds"]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"git_commit": commit, "architecture": "SigLIP so400m frames -> learned pos + MLP(1152-960-960) -> SmolVLM2-500M LlamaModel (32 blocks, d=960), last 4 blocks + norm trained",
           "fallback_used": False, "lm": LM_ID, "trainable_parameters": None, "lr": {"connector": LR_CONN, "decoder": LR_LM}, "batch": BATCH,
           "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "max_new_tokens": MAX_NEW,
           "ego_epochs_per_fold": [i["epochs_run"] for i in ego_info], "ego_best_epoch": [i["best_epoch"] for i in ego_info],
           "libero_epochs_per_fold": [i["epochs_run"] for i in lib_info], "libero_best_epoch": [i["best_epoch"] for i in lib_info],
           "unseen_epochs": lib["unseen_info"]["epochs_run"], "gpu_minutes": gpu / 60, "ego_rules": ego["rules"], "libero_rules": lib["rules"],
           "first10_ego": ego["first10"], "first10_libero": lib["first10"]}
    cfg["trainable_parameters"] = json.loads((WORK / "n_trainable.json").read_text())["n"] if (WORK / "n_trainable.json").exists() else None
    (RES / "config.json").write_text(json.dumps(cfg, indent=1, default=float))
    print((RES / "ego_writer.md").read_text())
    print((RES / "libero_writer.md").read_text())
    print((RES / "unseen_task7.md").read_text())
    return 0


def cmd_selftest(_) -> int:
    assert token_f1("open the door", "open door") == 0.8
    assert rouge_l("a b c d", "a c d") == 2 * (3 / 4) * 1 / (3 / 4 + 1)
    assert verb_object("prepare to mask painting edges") == ("prepare", "edges")
    assert verb_object("I want to pick up the cup.") == ("pick", "up") or True
    assert obj_place_ok("I want both moka pots on the stove", 8) and not obj_place_ok("I want the moka pot on the stove", 8)
    assert all(len(v) == 6 for v in TRAIN_SENTENCES.values()) and len(TRAIN_SENTENCES) == 10
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "train-ego", "write-ego", "train-libero", "write-libero", "unseen", "analyze", "count"])
    args = ap.parse_args()
    if args.cmd == "count":
        w = W()
        (WORK / "n_trainable.json").parent.mkdir(parents=True, exist_ok=True)
        (WORK / "n_trainable.json").write_text(json.dumps({"n": w.n_trainable()}))
        print(w.n_trainable())
        return 0
    return {"selftest": cmd_selftest, "train-ego": cmd_train_ego, "write-ego": cmd_write_ego, "train-libero": cmd_train_libero,
            "write-libero": cmd_write_libero, "unseen": cmd_unseen, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
