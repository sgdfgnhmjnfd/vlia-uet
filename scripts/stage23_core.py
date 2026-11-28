"""Stage 23 core: cache, features, models, stats, parts A/B, table."""
import collections
import csv
import json
import os
import time
import zipfile

import numpy as np

import stage23_breakfast as B

PS = [0.10, 0.25, 0.50, 1.00]
PNAMES = ["p10", "p25", "p50", "p100"]
NBOOT = 10000
FOLDS = os.path.join(B.RES, "folds")
NLAB = 48  # incl SIL (id 0)
CAM_PREF = ["cam01", "webcam01", "webcam02", "stereo01", "cam02"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mapping():
    with open(f"{B.META}/mapping.txt") as f:
        names = [l.split()[1] for l in f if l.strip()]
    assert names[0] == B.SIL
    return names, {n: i for i, n in enumerate(names)}


# ======================================================== pure feature functions
def pool_seconds(x, lab):
    """x (T,D), lab (T,) int -> seq1s (K,D) mean per 15-frame second (last partial kept), lab1s majority."""
    T = x.shape[0]
    K = (T + B.FPS - 1) // B.FPS
    seq = np.zeros((K, x.shape[1]), np.float32)
    l1 = np.zeros(K, np.int16)
    for k in range(K):
        a, b = k * B.FPS, min(k * B.FPS + B.FPS, T)
        seq[k] = x[a:b].mean(0)
        l1[k] = np.bincount(lab[a:b], minlength=NLAB).argmax()
    return seq, l1


def prefix_n(T, p):
    return int(min(T, max(B.FPS, round(p * T))))


def video_vectors(x, start, ps=PS):
    """Part A vectors for a prefix that begins at frame `start`.
    Returns S (1024, half1 mean of first 1 s), xs (2048, x_start), pm (len(ps),2048 prefix means), ns."""
    T = x.shape[0] - start
    xs = x[start:]
    S = xs[: B.FPS, :1024].mean(0)
    cs = np.cumsum(xs, 0, dtype=np.float64)
    ns = [prefix_n(T, p) for p in ps]
    pm = np.stack([cs[n - 1] / n for n in ns]).astype(np.float32)
    return S.astype(np.float32), xs[0].astype(np.float32), pm, np.array(ns)


def seg_feats(x, s, e):
    """SEG mean over [s,e); CTX half1 mean over [max(0,s-15), s); PRE mean over [0,s)."""
    seg = x[s:e].mean(0)
    ctx = x[max(0, s - B.FPS):s, :1024].mean(0) if s > 0 else np.zeros(1024, np.float32)
    pre = x[:s].mean(0) if s > 0 else np.zeros(x.shape[1], np.float32)
    return seg, ctx, pre


def hist_vec(labels):
    """labels: ordered list of non-SIL label ids (1..47) -> 141-d (bag normalised + last + second-last one-hot)."""
    v = np.zeros(141, np.float32)
    if len(labels):
        bc = np.bincount(np.asarray(labels) - 1, minlength=47).astype(np.float32)
        v[:47] = bc / bc.sum()
        v[47 + labels[-1] - 1] = 1
        if len(labels) > 1:
            v[94 + labels[-2] - 1] = 1
    return v


def hist_oracle_prefix(segs, n):
    """Part A: true non-SIL segments with start < n."""
    return hist_vec([int(l) for l, s, e in segs if s < n and l != 0])


def hist_oracle_before(segs, s0):
    """Part B: true non-SIL segments that end <= s0."""
    return hist_vec([int(l) for l, s, e in segs if e <= s0 and l != 0])


def smooth_causal(pred):
    out = np.empty_like(pred)
    for k in range(len(pred)):
        w = pred[max(0, k - 2):k + 1]
        out[k] = np.bincount(w, minlength=NLAB).argmax()
    return out


def runs_nonsil(lbl):
    r = []
    prev = -1
    for l in lbl:
        if l != prev:
            if l != 0:
                r.append(int(l))
            prev = l
    return r


def hist_pred_upto(sm, nframes):
    """Use smoothed per-second predictions of seconds k with 15k+15 <= nframes."""
    K = nframes // B.FPS
    return hist_vec(runs_nonsil(sm[:K]))


# ======================================================== cache
def cache_path(vid):
    return f"{B.CACHE}/{vid}.npz"


def cmd_cache(args):
    names, idx = mapping()
    vids = B.list_videos()
    z = zipfile.ZipFile(B.ZIP)
    t0 = time.time()
    done = 0
    for i, v in enumerate(vids):
        if os.path.exists(cache_path(v)):
            continue
        x = B.load_feat(z, v)
        lab = np.array([idx[l] for l in B.read_gt(v)], np.int64)
        T = min(len(x), len(lab))
        x, lab = x[:T], lab[:T]
        seq, l1 = pool_seconds(x, lab)
        segs = np.array([(idx[l], s, e) for l, s, e in B.segments([names[j] for j in lab])], np.int32)
        sf = [seg_feats(x, s, e) for _, s, e in segs]
        S0, x0, pm0, n0 = video_vectors(x, 0)
        nz = np.nonzero(lab != 0)[0]
        f0 = int(nz[0]) if len(nz) and T - nz[0] >= B.FPS else 0
        Sn, xn, pmn, nn = video_vectors(x, f0)
        tmp = cache_path(v) + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, seq1s=seq.astype(np.float16), lab1s=l1, segs=segs,
                     seg_mean=np.stack([a for a, _, _ in sf]).astype(np.float16),
                     ctx_h1=np.stack([b for _, b, _ in sf]).astype(np.float16),
                     pre=np.stack([c for _, _, c in sf]).astype(np.float16),
                     S0=S0, x0=x0, pm0=pm0, n0=n0, f0=f0, Sn=Sn, xn=xn, pmn=pmn, nn=nn, T=T)
        os.replace(tmp, cache_path(v))
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            log(f"cache {i+1}/{len(vids)} done, eta {el/done*(len(vids)-i-1)/60:.1f} min")
    log("cache complete", len(vids))


# ======================================================== models
def fit_linear(Xtr, ytr, gtr, Xte, grid=(0.01, 0.1, 1, 10), return_model=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def mk(C):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)

    gkf = GroupKFold(n_splits=3)
    cache = {}

    def score(C):
        if C in cache:
            return cache[C]
        sc = []
        for a, b in gkf.split(Xtr, ytr, gtr):
            ss = StandardScaler().fit(Xtr[a])
            m = mk(C).fit(ss.transform(Xtr[a]), ytr[a])
            sc.append(balanced_accuracy_score(ytr[b], m.predict(ss.transform(Xtr[b]))))
        cache[C] = float(np.mean(sc))
        return cache[C]

    grid = list(grid)
    best = max(grid, key=lambda C: (score(C), -C))
    ext = None
    if best == grid[0]:
        ext = grid[0] * 0.1
    elif best == grid[-1]:
        ext = grid[-1] * 10
    if ext is not None:
        best = max(grid + [ext], key=lambda C: (score(C), -C))
    ss = StandardScaler().fit(Xtr)
    m = mk(best).fit(ss.transform(Xtr), ytr)
    yhat = m.predict(ss.transform(Xte))
    info = dict(C=best, cv=cache, extended=ext)
    if return_model:
        return yhat, info, ss, m
    return yhat, info


def boot_weights(n_people, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_people, np.ones(n_people) / n_people, size=nboot).astype(np.float64)


def person_tables(pid, y, yhat, K):
    """pid int index 0..P-1 -> tot (P,K), cor (P,K)."""
    P = pid.max() + 1
    tot = np.zeros((P, K))
    cor = np.zeros((P, K))
    np.add.at(tot, (pid, y), 1)
    np.add.at(cor, (pid, y), (y == yhat).astype(float))
    return tot, cor


def balacc_boot(W, tot, cor):
    """W (B,P) -> (B,) balanced accuracy (mean recall over classes present in the resample)."""
    tb = W @ tot
    cb = W @ cor
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cb / tb
    return np.nanmean(r, 1)


def metrics_point(y, yhat, K):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return dict(acc=float((y == yhat).mean()), bal_acc=float(balanced_accuracy_score(y, yhat)),
                macro_f1=float(f1_score(y, yhat, average="macro", labels=list(range(K)), zero_division=0)))


# ======================================================== loading
def meta_all():
    vids = B.list_videos()
    m = {v: B.parse_vid(v) for v in vids}
    dishes = sorted({d for _, _, d in m.values()})
    people = sorted({p for p, _, _ in m.values()})
    return vids, m, dishes, people


def load_npz(v):
    return np.load(cache_path(v))


def pca_pipeline(vids_fit, nmax, seed):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    lens = []
    for v in vids_fit:
        with load_npz(v) as d:
            lens.append(d["seq1s"].shape[0])
    tot = sum(lens)
    take = min(nmax, tot)
    sel = np.sort(rng.choice(tot, take, replace=False))
    X = np.zeros((take, 2048), np.float32)
    off = 0
    j = 0
    for v, L in zip(vids_fit, lens):
        idx = sel[(sel >= off) & (sel < off + L)] - off
        if len(idx):
            with load_npz(v) as d:
                X[j:j + len(idx)] = d["seq1s"][idx]
            j += len(idx)
        off += L
    ss = StandardScaler().fit(X)
    pca = PCA(256, random_state=0, svd_solver="randomized").fit(ss.transform(X))
    return ss, pca


def project(vids, ss, pca):
    out = {}
    lab = {}
    for v in vids:
        with load_npz(v) as d:
            out[v] = pca.transform(ss.transform(d["seq1s"].astype(np.float32))).astype(np.float32)
            lab[v] = d["lab1s"].astype(np.int64)
    return out, lab


def train_frame_clf(Z, L, vids, nmax=60000, seed=0):
    from sklearn.linear_model import LogisticRegression
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X = np.concatenate([Z[v] for v in vids])
    y = np.concatenate([L[v] for v in vids])
    rng = np.random.default_rng(seed)
    if len(X) > nmax:
        i = rng.choice(len(X), nmax, replace=False)
        X, y = X[i], y[i]
    return LogisticRegression(C=1.0, max_iter=300).fit(X, y)


def frame_preds_for_split(k, sp, m):
    """Returns (smoothed preds dict for all videos, Zfull dict, frame acc info). Cached on disk."""
    fp = f"{B.CACHE}/framepred_split{k}.npz"
    zp = f"{B.CACHE}/pca_split{k}.npz"
    if os.path.exists(fp) and os.path.exists(zp):
        d = np.load(fp, allow_pickle=True)
        zd = np.load(zp, allow_pickle=True)
        return d["sm"].item(), zd["Z"].item(), d["info"].item()
    tr, te = sp[k]["train"], sp[k]["test"]
    t0 = time.time()
    ss, pca = pca_pipeline(tr, 40000, 0)
    Z, L = project(tr + te, ss, pca)
    clf = train_frame_clf(Z, L, tr)
    sm = {}
    raw_acc = []
    for v in te:
        pr = clf.predict(Z[v])
        raw_acc.append((pr == L[v]).mean())
        sm[v] = smooth_causal(pr.astype(np.int64))
    log(f"split{k} full frame clf done {time.time()-t0:.0f}s, test sec-acc {np.mean(raw_acc):.3f}")
    # cross-fit for train videos
    trp = sorted({m[v][0] for v in tr})
    grp = {p: i % 3 for i, p in enumerate(trp)}
    for g in range(3):
        fit_v = [v for v in tr if grp[m[v][0]] != g]
        pred_v = [v for v in tr if grp[m[v][0]] == g]
        ssg, pcag = pca_pipeline(fit_v, 40000, 0)
        Zg, Lg = project(fit_v + pred_v, ssg, pcag)
        clfg = train_frame_clf(Zg, Lg, fit_v)
        for v in pred_v:
            sm[v] = smooth_causal(clfg.predict(Zg[v]).astype(np.int64))
        del Zg, Lg
        log(f"split{k} crossfit group {g} done {time.time()-t0:.0f}s")
    sm_acc = float(np.mean([(sm[v] == L[v]).mean() for v in te]))
    info = dict(test_sec_acc_raw=float(np.mean(raw_acc)), test_sec_acc_smoothed=sm_acc, seconds=time.time() - t0)
    np.savez(fp + ".tmp.npz", sm=np.array(sm, dtype=object), info=np.array(info, dtype=object))
    os.replace(fp + ".tmp.npz", fp)
    np.savez(zp + ".tmp.npz", Z=np.array(Z, dtype=object))
    os.replace(zp + ".tmp.npz", zp)
    return sm, Z, info


# ======================================================== Part A
ROWS_A_LIN = ["S", "M-delta", "M-flow", "S+M", "H-oracle", "H-pred", "S+M+H-pred"]


def featsA(d, sm, row, pi, variant="0"):
    if variant == "0":
        S, x0, pm, ns = d["S0"], d["x0"], d["pm0"], d["n0"]
        start = 0
    else:
        S, x0, pm, ns = d["Sn"], d["xn"], d["pmn"], d["nn"]
        start = int(d["f0"])
    n = int(ns[pi])
    md = pm[pi] - x0
    if row == "S":
        return S
    if row == "M-delta":
        return md
    if row == "M-flow":
        return pm[pi][1024:]
    if row == "S+M":
        return np.concatenate([S, md])
    if row == "H-oracle":
        return hist_oracle_prefix(d["segs"], start + n)
    if row == "H-pred":
        return hist_pred_upto(sm, start + n)
    if row == "S+M+H-pred":
        return np.concatenate([S, md, hist_pred_upto(sm, start + n)])
    raise KeyError(row)


def save_fold(name, obj):
    os.makedirs(FOLDS, exist_ok=True)
    with open(f"{FOLDS}/{name}.json.tmp", "w") as f:
        json.dump(obj, f)
    os.replace(f"{FOLDS}/{name}.json.tmp", f"{FOLDS}/{name}.json")


def have_fold(name):
    return os.path.exists(f"{FOLDS}/{name}.json")


def train_T(Z, vids_tr, y_tr, people_tr, vids_te, ns_tr, ns_te, seed, ch=128, K=10):
    import torch
    import torch.nn as nn
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    up = sorted(set(people_tr))
    val_p = set(rng.choice(up, max(1, round(0.2 * len(up))), replace=False))
    is_val = np.array([p in val_p for p in people_tr])

    def seqs(vs, ns):
        return [torch.from_numpy(Z[v][: max(1, n // B.FPS)]) for v, n in zip(vs, ns)]

    S_tr = seqs(vids_tr, ns_tr)
    S_te = seqs(vids_te, ns_te)
    y_tr = np.asarray(y_tr)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv1d(256, ch, 3, padding=1)
            s.c2 = nn.Conv1d(ch, ch, 3, padding=1)
            s.fc = nn.Linear(ch, K)

        def forward(s, x, mask):
            h = torch.relu(s.c2(torch.relu(s.c1(x))))
            h = (h * mask[:, None]).sum(2) / mask.sum(1, keepdim=True)
            return s.fc(h)

    def batchify(lst):
        L = max(t.shape[0] for t in lst)
        x = torch.zeros(len(lst), 256, L)
        mk = torch.zeros(len(lst), L)
        for i, t in enumerate(lst):
            x[i, :, : t.shape[0]] = t.T
            mk[i, : t.shape[0]] = 1
        return x, mk

    def predict(net, lst):
        net.eval()
        out = []
        order = np.argsort([t.shape[0] for t in lst])
        res = np.zeros(len(lst), np.int64)
        with torch.no_grad():
            for i in range(0, len(lst), 64):
                ids = order[i:i + 64]
                x, mk = batchify([lst[j] for j in ids])
                res[ids] = net(x, mk).argmax(1).numpy()
        return res

    from sklearn.metrics import balanced_accuracy_score
    tr_idx = np.nonzero(~is_val)[0]
    va_idx = np.nonzero(is_val)[0]
    cw = np.bincount(y_tr[tr_idx], minlength=K).astype(np.float32)
    cw = len(tr_idx) / (K * np.maximum(cw, 1))
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(cw))
    best, best_state, bad = -1, None, 0
    for ep in range(30):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 32):
            ids = perm[i:i + 32]
            x, mk = batchify([S_tr[j] for j in ids])
            opt.zero_grad()
            lossf(net(x, mk), torch.from_numpy(y_tr[ids])).backward()
            opt.step()
        va = balanced_accuracy_score(y_tr[va_idx], predict(net, [S_tr[j] for j in va_idx]))
        if va > best:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    return predict(net, S_te), dict(val_bal_acc=best, epochs=ep + 1)


def cmd_partA(args):
    vids, m, dishes, people = meta_all()
    dix = {d: i for i, d in enumerate(dishes)}
    sp = B.read_splits()
    for k in [int(s) for s in args.splits.split(",")]:
        tr, te = sp[k]["train"], sp[k]["test"]
        assert not ({m[v][0] for v in tr} & {m[v][0] for v in te})
        t0 = time.time()
        sm, Z, finfo = frame_preds_for_split(k, sp, m)
        save_fold(f"frameclf_split{k}", finfo)
        D = {v: dict(load_npz(v)) for v in tr + te}
        ytr = np.array([dix[m[v][2]] for v in tr])
        yte = np.array([dix[m[v][2]] for v in te])
        gtr = np.array([m[v][0] for v in tr])
        for variant, rows in (("0", ROWS_A_LIN), ("ns", ["S", "M-delta", "M-flow", "S+M"])):
            for pi, pn in enumerate(PNAMES):
                for row in rows:
                    name = f"A_{row}_{variant}_{pn}_split{k}"
                    if have_fold(name):
                        continue
                    Xtr = np.stack([featsA(D[v], sm[v], row, pi, variant) for v in tr])
                    Xte = np.stack([featsA(D[v], sm[v], row, pi, variant) for v in te])
                    yhat, info = fit_linear(Xtr, ytr, gtr, Xte)
                    save_fold(name, dict(vids=te, y=yte.tolist(), yhat=yhat.tolist(), info=info,
                                         majority=int(np.bincount(ytr).argmax())))
                    log(name, f"bal {metrics_point(yte, yhat, 10)['bal_acc']:.3f} C={info['C']}")
        # label shuffle check
        name = f"A_S+M-shuffled_0_p25_split{k}"
        if not have_fold(name):
            rng = np.random.default_rng(0)
            ysh = ytr.copy()
            for p in sorted(set(gtr)):
                ii = np.nonzero(gtr == p)[0]
                ysh[ii] = ysh[rng.permutation(ii)]
            Xtr = np.stack([featsA(D[v], sm[v], "S+M", 1) for v in tr])
            Xte = np.stack([featsA(D[v], sm[v], "S+M", 1) for v in te])
            yhat, info = fit_linear(Xtr, ysh, gtr, Xte)
            save_fold(name, dict(vids=te, y=yte.tolist(), yhat=yhat.tolist(), info=info, majority=0))
            log(name, f"bal {metrics_point(yte, yhat, 10)['bal_acc']:.3f}")
        log(f"split{k} linear done {time.time()-t0:.0f}s")
        # temporal model
        t1 = time.time()
        for pi, pn in enumerate(PNAMES):
            for seed in (0, 1, 2):
                name = f"A_T_0_{pn}_split{k}_seed{seed}"
                if have_fold(name):
                    continue
                ns_tr = [int(D[v]["n0"][pi]) for v in tr]
                ns_te = [int(D[v]["n0"][pi]) for v in te]
                yhat, info = train_T(Z, tr, ytr, list(gtr), te, ns_tr, ns_te, seed)
                info["seconds"] = time.time() - t1
                save_fold(name, dict(vids=te, y=yte.tolist(), yhat=yhat.tolist(), info=info, majority=0))
                log(name, f"bal {metrics_point(yte, yhat, 10)['bal_acc']:.3f} {info}")
        log(f"split{k} T done {time.time()-t1:.0f}s; split total {time.time()-t0:.0f}s")
        del D, Z, sm


# ======================================================== Part B
ROWS_B = ["SEG", "SEG-half1", "SEG-half2", "CTX-S", "HIST-oracle", "HIST-pred", "PRE", "SEG+HIST-pred", "SEG+CTX-S"]


def featB(d, sm, j, row):
    segs = d["segs"]
    s = int(segs[j, 1])
    seg = d["seg_mean"][j].astype(np.float32)
    ctx = d["ctx_h1"][j].astype(np.float32)
    if row == "SEG":
        return seg
    if row == "SEG-half1":
        return seg[:1024]
    if row == "SEG-half2":
        return seg[1024:]
    if row == "CTX-S":
        return ctx
    if row == "HIST-oracle":
        return hist_oracle_before(segs, s)
    if row == "HIST-pred":
        return hist_pred_upto(sm, s)
    if row == "PRE":
        return d["pre"][j].astype(np.float32)
    if row == "SEG+HIST-pred":
        return np.concatenate([seg, hist_pred_upto(sm, s)])
    if row == "SEG+CTX-S":
        return np.concatenate([seg, ctx])
    raise KeyError(row)


def segB_index(vids, m, astar, idx):
    """list of (action, vid, seg_row_index, dish)."""
    out = []
    for v in vids:
        dish = m[v][2]
        with load_npz(v) as d:
            segs = d["segs"]
        for c in astar:
            if dish not in c["dishes"]:
                continue
            ai = idx[c["action"]]
            for j, (l, s, e) in enumerate(segs):
                if l == ai and e - s >= B.FPS:
                    out.append((c["action"], v, j, dish))
    return out


def cmd_partB(args):
    vids, m, dishes, people = meta_all()
    names, idx = mapping()
    astar = json.load(open(f"{B.RES}/astar.json"))
    sp = B.read_splits()
    for k in [int(s) for s in args.splits.split(",")]:
        tr, te = sp[k]["train"], sp[k]["test"]
        sm, Z, _ = frame_preds_for_split(k, sp, m)
        del Z
        D = {v: dict(load_npz(v)) for v in tr + te}
        itr = segB_index(tr, m, astar, idx)
        ite = segB_index(te, m, astar, idx)
        for c in astar:
            a = c["action"]
            cix = {d: i for i, d in enumerate(c["dishes"])}
            Rtr = [r for r in itr if r[0] == a]
            Rte = [r for r in ite if r[0] == a]
            ytr = np.array([cix[r[3]] for r in Rtr])
            yte = np.array([cix[r[3]] for r in Rte])
            gtr = np.array([m[r[1]][0] for r in Rtr])
            for row in ROWS_B:
                name = f"B_{row}_{a}_split{k}"
                if have_fold(name):
                    continue
                if len(Rte) == 0:
                    save_fold(name, dict(vids=[], segj=[], y=[], yhat=[], info={"empty_test": True}, K=len(cix),
                                         majority=int(np.bincount(ytr, minlength=len(cix)).argmax())))
                    log(name, "no test segments in this split -> skipped")
                    continue
                Xtr = np.stack([featB(D[r[1]], sm[r[1]], r[2], row) for r in Rtr])
                Xte = np.stack([featB(D[r[1]], sm[r[1]], r[2], row) for r in Rte])
                yhat, info = fit_linear(Xtr, ytr, gtr, Xte)
                save_fold(name, dict(vids=[r[1] for r in Rte], segj=[r[2] for r in Rte], y=yte.tolist(),
                                     yhat=yhat.tolist(), info=info, K=len(cix),
                                     majority=int(np.bincount(ytr, minlength=len(cix)).argmax())))
                log(name, f"n_tr {len(Rtr)} n_te {len(Rte)} bal {metrics_point(yte, yhat, len(cix))['bal_acc']:.3f}")
        del D, sm


# ======================================================== selftest
def cmd_selftest(args):
    rng = np.random.default_rng(0)
    # splits: no person in both
    vids, m, dishes, people = meta_all()
    sp = B.read_splits()
    for k, d in sp.items():
        assert not ({m[v][0] for v in d["train"]} & {m[v][0] for v in d["test"]}), k
    # M-delta = 0 for constant frames; S independent of p
    x = np.tile(rng.normal(size=(1, 2048)).astype(np.float32), (300, 1))
    S, x0, pm, ns = video_vectors(x, 0)
    assert np.allclose(pm - x0, 0, atol=1e-5)
    x = rng.normal(size=(300, 2048)).astype(np.float32)
    S1, _, _, _ = video_vectors(x, 0, ps=[0.1])
    S2, _, _, _ = video_vectors(x, 0, ps=[1.0])
    assert np.array_equal(S1, S2)
    S3, xs3, _, _ = video_vectors(x, 40)
    assert np.allclose(S3, x[40:55, :1024].mean(0)) and np.array_equal(xs3, x[40])
    # prefix mean uses only first n frames
    _, _, pm, ns = video_vectors(x, 0, ps=[0.25])
    assert np.allclose(pm[0], x[: ns[0]].mean(0), atol=1e-5)
    # Part B leakage: frames >= s -> NaN must not affect CTX/PRE; frames outside [s,e) NaN must not affect SEG
    s, e = 100, 160
    xl = x.copy(); xl[s:] = np.nan
    _, ctx, pre = seg_feats(xl, s, e)
    assert np.isfinite(ctx).all() and np.isfinite(pre).all()
    xo = x.copy(); xo[:s] = np.nan; xo[e:] = np.nan
    seg, _, _ = seg_feats(xo, s, e)
    assert np.isfinite(seg).all()
    # HIST-pred: changing predictions of seconds that overlap [s, ...) must not change features
    pred = rng.integers(0, NLAB, 30)
    h1 = hist_pred_upto(smooth_causal(pred), s)
    pred2 = pred.copy(); pred2[s // B.FPS:] = rng.integers(0, NLAB, 30 - s // B.FPS)
    h2 = hist_pred_upto(smooth_causal(pred2), s)
    assert np.array_equal(h1, h2)
    # causal smoothing
    sm1 = smooth_causal(pred); sm2 = smooth_causal(pred2)
    assert np.array_equal(sm1[: s // B.FPS], sm2[: s // B.FPS])
    # HIST-oracle excludes segment a and later
    segs = np.array([(0, 0, 50), (3, 50, 100), (5, 100, 160), (7, 160, 300)])
    h = hist_oracle_before(segs, 100)
    assert h[47 + 3 - 1] == 1 and h[:47].sum() == 1 and h[5 - 1] == 0 and h[7 - 1] == 0
    # H-oracle prefix: start < n
    h = hist_oracle_prefix(segs, 101)
    assert h[3 - 1] > 0 and h[5 - 1] > 0 and h[7 - 1] == 0
    # scaler/PCA/C only on train: fit_linear scaler mean equals train mean even if test is shifted
    Xtr = rng.normal(size=(120, 5)); ytr = (Xtr[:, 0] > 0).astype(int); gtr = np.repeat(np.arange(12), 10)
    Xte = rng.normal(size=(30, 5)) + 100
    _, info, ss, mdl = fit_linear(Xtr, ytr, gtr, Xte, return_model=True)
    assert np.allclose(ss.mean_, Xtr.mean(0))
    # bootstrap by person
    W = boot_weights(52, nboot=200)
    assert W.shape == (200, 52) and np.all(W.sum(1) == 52)
    pid = np.repeat(np.arange(52), 3); y = rng.integers(0, 4, 156)
    tot, cor = person_tables(pid, y, y, 4)
    assert np.allclose(balacc_boot(W, tot, cor), 1.0)
    # pooling
    lab = np.zeros(40, np.int64); lab[20:] = 2
    seq, l1 = pool_seconds(x[:40], lab)
    assert seq.shape[0] == 3 and list(l1) == [0, 2, 2]
    print("selftest OK")


# ======================================================== table
def collect(prefix):
    out = {}
    for f in os.listdir(FOLDS):
        if f.startswith(prefix):
            out[f[:-5]] = json.load(open(f"{FOLDS}/{f}"))
    return out


def cmd_table(args):
    from sklearn.metrics import confusion_matrix
    vids, m, dishes, people = meta_all()
    pix = {p: i for i, p in enumerate(people)}
    W = boot_weights(len(people))
    A = collect("A_")
    Bf = collect("B_")
    lines = []
    rowsA = []
    bootA = {}

    def pooled(keys):
        y, yh, pid, vv = [], [], [], []
        for kk in keys:
            d = A[kk] if kk in A else Bf[kk]
            y += d["y"]; yh += d["yhat"]; vv += d["vids"]
            pid += [pix[m[v][0]] for v in d["vids"]]
        return np.array(y), np.array(yh), np.array(pid), vv

    # one-camera subset
    rec_cam = {}
    for v in vids:
        p, c, d = m[v]
        r = (p, d)
        if r not in rec_cam or CAM_PREF.index(c) < CAM_PREF.index(rec_cam[r]):
            rec_cam[r] = c
    onecam = {v for v in vids if rec_cam[(m[v][0], m[v][2])] == m[v][1]}

    def addA(rowname, variant, pn, keys, K=10, subset=None, seedtag=""):
        y, yh, pid, vv = pooled(keys)
        if subset is not None:
            mk = np.array([v in subset for v in vv]); y, yh, pid = y[mk], yh[mk], pid[mk]
        tot, cor = person_tables(pid, y, yh, K)
        bb = balacc_boot(W, tot, cor)
        mp = metrics_point(y, yh, K)
        maj = np.mean([A[kk]["majority"] == yy for kk in keys for yy in A[kk]["y"]]) if subset is None else np.nan
        r = dict(row=rowname, variant=variant, p=pn, n=len(y), **mp, ci_lo=float(np.percentile(bb, 2.5)),
                 ci_hi=float(np.percentile(bb, 97.5)), chance=1 / K, majority_acc=float(maj), seed=seedtag)
        rowsA.append(r)
        return bb

    for variant in ("0", "ns"):
        for pn in PNAMES:
            for row in (ROWS_A_LIN if variant == "0" else ["S", "M-delta", "M-flow", "S+M"]):
                keys = [f"A_{row}_{variant}_{pn}_split{k}" for k in range(1, 5)]
                if all(kk in A for kk in keys):
                    bootA[(row, variant, pn)] = addA(row, variant, pn, keys)
            if variant == "0":
                seeds = []
                for seed in (0, 1, 2):
                    keys = [f"A_T_0_{pn}_split{k}_seed{seed}" for k in range(1, 5)]
                    if all(kk in A for kk in keys):
                        seeds.append(addA("T", "0", pn, keys, seedtag=str(seed)))
                if len(seeds) == 3:
                    bb = np.mean(seeds, 0)
                    bootA[("T", "0", pn)] = bb
                    last = rowsA[-3:]
                    rowsA.append(dict(row="T", variant="0", p=pn, n=last[0]["n"],
                                      acc=np.mean([r["acc"] for r in last]), bal_acc=np.mean([r["bal_acc"] for r in last]),
                                      macro_f1=np.mean([r["macro_f1"] for r in last]),
                                      ci_lo=float(np.percentile(bb, 2.5)), ci_hi=float(np.percentile(bb, 97.5)),
                                      chance=0.1, majority_acc=np.nan, seed="mean3"))
    keys = [f"A_S+M-shuffled_0_p25_split{k}" for k in range(1, 5)]
    if all(kk in A for kk in keys):
        addA("S+M-shuffled", "0", "p25", keys)
    for row in ("S", "M-delta", "S+M"):
        keys = [f"A_{row}_0_p25_split{k}" for k in range(1, 5)]
        if all(kk in A for kk in keys):
            addA(row + "@onecam", "0", "p25", keys, subset=onecam)

    with open(f"{B.RES}/accuracy_partA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rowsA[0].keys()))
        w.writeheader(); w.writerows(rowsA)

    def get(row, pn, variant="0", seed=""):
        for r in rowsA:
            if r["row"] == row and r["p"] == pn and r["variant"] == variant and r["seed"] == (seed or r["seed"]) and \
                    (row != "T" or r["seed"] == "mean3"):
                return r

    def diff(a, b, pn):
        da = bootA[(a, "0", pn)] - bootA[(b, "0", pn)]
        pa = get(a, pn)["bal_acc"] - get(b, pn)["bal_acc"]
        return pa, np.percentile(da, 2.5), np.percentile(da, 97.5)

    # Part A summary
    lines.append("# Stage 23 Breakfast: summary\n")
    lines.append("Balanced accuracy, pooled over 4 official splits, 95% CI by person bootstrap (10,000). Chance Part A = 0.10.")
    lines.append("half1/half2 = first/second 1024 dims of I3D vector (no source for RGB/flow order).\n")
    lines.append("## Part A: early dish prediction (from frame 0)\n")
    lines.append("| row | " + " | ".join(PNAMES) + " |")
    lines.append("|---|" + "---|" * len(PNAMES))
    for row in ["S", "M-delta", "M-flow", "S+M", "T", "H-oracle", "H-pred", "S+M+H-pred"]:
        cells = []
        for pn in PNAMES:
            r = get(row, pn)
            cells.append(f"{r['bal_acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r else "-")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines.append("\nT seeds: " + "; ".join(
        f"{pn}: " + ", ".join(f"{r['bal_acc']:.3f}" for r in rowsA if r["row"] == "T" and r["p"] == pn and r["seed"] in "012")
        for pn in PNAMES))
    lines.append("\n### Part A from first non-SIL frame (descriptive)\n")
    lines.append("| row | " + " | ".join(PNAMES) + " |")
    lines.append("|---|" + "---|" * len(PNAMES))
    for row in ["S", "M-delta", "M-flow", "S+M"]:
        cells = []
        for pn in PNAMES:
            r = get(row, pn, "ns")
            cells.append(f"{r['bal_acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r else "-")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines.append("\n### Checks (p25)\n")
    for rr in rowsA:
        if rr["row"].endswith("shuffled") or rr["row"].endswith("@onecam"):
            lines.append(f"- {rr['row']}: {rr['bal_acc']:.3f} [{rr['ci_lo']:.3f}, {rr['ci_hi']:.3f}] n={rr['n']}")
    for row in ("S", "S+M"):
        r = get(row, "p25")
        lines.append(f"- majority baseline acc ({row} folds): {r['majority_acc']:.3f}")

    lines.append("\n### Hypotheses Part A (p25)\n")
    rS = get("S", "p25")
    ha1 = rS["bal_acc"] >= 0.80
    lines.append(f"- HA-1: S = {rS['bal_acc']:.3f} -> {'>= 0.80: scene already sufficient' if ha1 else '< 0.80: scene alone not sufficient'}")
    d, lo, hi = diff("S+M", "S", "p25")
    lines.append(f"- HA-2: (S+M) - S = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if lo > 0 else 'FAIL'}")
    rM = get("M-delta", "p25")
    lines.append(f"- HA-3: M-delta = {rM['bal_acc']:.3f} [{rM['ci_lo']:.3f}, {rM['ci_hi']:.3f}] vs 0.10 -> {'PASS' if rM['ci_lo'] > 0.10 else 'FAIL'}")
    if ("T", "0", "p25") in bootA:
        d, lo, hi = diff("T", "S+M", "p25")
        lines.append(f"- HA-4: T - (S+M) = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if lo > 0 and d >= 0.03 else 'FAIL'}")
    for pn in PNAMES:
        if ("H-oracle", "0", pn) in bootA:
            d, lo, hi = diff("H-oracle", "H-pred", pn)
            lines.append(f"- (descr.) H-oracle - H-pred {pn}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    fc = [json.load(open(f"{FOLDS}/frameclf_split{k}.json")) for k in range(1, 5) if os.path.exists(f"{FOLDS}/frameclf_split{k}.json")]
    if fc:
        lines.append(f"- (descr.) frame classifier per-second acc on test (48 cls incl SIL): raw {np.mean([f['test_sec_acc_raw'] for f in fc]):.3f}, smoothed {np.mean([f['test_sec_acc_smoothed'] for f in fc]):.3f}")

    # per person + confusion
    pp = []
    for row in ("S", "M-delta", "S+M", "H-pred"):
        y, yh, pid, _ = pooled([f"A_{row}_0_p25_split{k}" for k in range(1, 5)])
        for p in sorted(set(pid)):
            mk = pid == p
            pp.append(dict(row=row, p="p25", person=people[p], n=int(mk.sum()), acc=float((y[mk] == yh[mk]).mean())))
    with open(f"{B.RES}/per_person.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pp[0].keys())); w.writeheader(); w.writerows(pp)
    y, yh, _, _ = pooled([f"A_S+M_0_p25_split{k}" for k in range(1, 5)])
    cm = confusion_matrix(y, yh, labels=list(range(10)))
    with open(f"{B.RES}/confusion_partA_p25.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + dishes)
        for i, d in enumerate(dishes):
            w.writerow([d] + cm[i].tolist())

    # Part B
    astar = json.load(open(f"{B.RES}/astar.json"))
    rowsB = []
    bootB = {}
    for row in ROWS_B:
        per_a = []
        for c in astar:
            a = c["action"]
            keys = [f"B_{row}_{a}_split{k}" for k in range(1, 5)]
            if not all(kk in Bf for kk in keys):
                continue
            K = c["K"]
            y, yh, pid, _ = pooled(keys)
            tot, cor = person_tables(pid, y, yh, K)
            # pad tables to all people (absent people contribute zeros)
            tot2 = np.zeros((len(people), K)); cor2 = np.zeros((len(people), K))
            tot2[: tot.shape[0]] = tot; cor2[: cor.shape[0]] = cor
            bb = balacc_boot(W, tot2, cor2)
            per_a.append(bb)
            mp = metrics_point(y, yh, K)
            maj = np.mean([Bf[kk]["majority"] == yy for kk in keys for yy in Bf[kk]["y"]])
            rowsB.append(dict(row=row, action=a, n=len(y), **mp, ci_lo=float(np.percentile(bb, 2.5)),
                              ci_hi=float(np.percentile(bb, 97.5)), chance=1 / K, majority_acc=float(maj), seed=""))
        if len(per_a) == len(astar):
            bb = np.mean(per_a, 0)
            bootB[row] = bb
            sub = [r for r in rowsB if r["row"] == row]
            rowsB.append(dict(row=row, action="MACRO", n=sum(r["n"] for r in sub),
                              acc=np.mean([r["acc"] for r in sub]), bal_acc=np.mean([r["bal_acc"] for r in sub]),
                              macro_f1=np.mean([r["macro_f1"] for r in sub]),
                              ci_lo=float(np.percentile(bb, 2.5)), ci_hi=float(np.percentile(bb, 97.5)),
                              chance=np.mean([1 / c["K"] for c in astar]), majority_acc=np.nan, seed=""))
    if rowsB:
        with open(f"{B.RES}/accuracy_partB.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsB[0].keys())); w.writeheader(); w.writerows(rowsB)
        acts = [c["action"] for c in astar] + ["MACRO"]
        lines.append("\n## Part B: same action, different intent (balanced accuracy)\n")
        lines.append("| row | " + " | ".join(f"{a} (K={c['K']})" for a, c in zip(acts, astar + [None]) if c) + " | MACRO |")
        lines.append("|---|" + "---|" * len(acts))
        lines.append("| chance | " + " | ".join(f"{1/c['K']:.3f}" for c in astar) + f" | {np.mean([1/c['K'] for c in astar]):.3f} |")
        lines.append("| n test segs | " + " | ".join(str(next(r['n'] for r in rowsB if r['row']=='SEG' and r['action']==a)) for a in acts[:-1]) + " | |")
        for row in ROWS_B:
            cells = []
            for a in acts:
                r = next((r for r in rowsB if r["row"] == row and r["action"] == a), None)
                cells.append((f"{r['bal_acc']:.3f}" + (f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if a == "MACRO" else "")) if r else "-")
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        if all(r in bootB for r in ("SEG", "HIST-pred", "CTX-S")):
            ch = np.mean([1 / c["K"] for c in astar])
            seg = next(r for r in rowsB if r["row"] == "SEG" and r["action"] == "MACRO")
            hb1 = seg["ci_lo"] > ch
            dd = bootB["HIST-pred"] - bootB["SEG"]
            pd = np.mean(bootB["HIST-pred"]) * 0  # placeholder
            pt = lambda row: next(r for r in rowsB if r["row"] == row and r["action"] == "MACRO")["bal_acc"]
            d2 = pt("HIST-pred") - pt("SEG"); lo2, hi2 = np.percentile(dd, [2.5, 97.5])
            d3b = bootB["HIST-pred"] - bootB["CTX-S"]
            d3 = pt("HIST-pred") - pt("CTX-S"); lo3, hi3 = np.percentile(d3b, [2.5, 97.5])
            hb2, hb3 = lo2 > 0, lo3 > 0
            lines.append("\n### Hypotheses Part B (macro over actions)\n")
            lines.append(f"- HB-1: SEG = {seg['bal_acc']:.3f} [{seg['ci_lo']:.3f}, {seg['ci_hi']:.3f}] vs chance {ch:.3f} -> {'PASS' if hb1 else 'FAIL'}")
            lines.append(f"- HB-2: HIST-pred - SEG = {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}] -> {'PASS' if hb2 else 'FAIL'}")
            lines.append(f"- HB-3: HIST-pred - CTX-S = {d3:+.3f} [{lo3:+.3f}, {hi3:+.3f}] -> {'PASS' if hb3 else 'FAIL'}")
            for a_, b_ in (("HIST-oracle", "HIST-pred"), ("SEG-half2", "SEG-half1"), ("SEG+HIST-pred", "SEG"), ("PRE", "SEG")):
                if a_ in bootB and b_ in bootB:
                    db = bootB[a_] - bootB[b_]
                    lines.append(f"- (descr.) {a_} - {b_} = {pt(a_)-pt(b_):+.3f} [{np.percentile(db,2.5):+.3f}, {np.percentile(db,97.5):+.3f}]")
            if hb2 and hb3:
                concl = "1. On Breakfast, a single action is not enough to know the intent; the preceding action sequence predicts the intent better than both the action segment and the scene just before it."
            elif hb2:
                concl = "2. The action sequence helps more than a single action, but not more than the scene just before."
            else:
                concl = "3. Not enough evidence that the action sequence predicts intent better than a single action."
            lines.append(f"\n**Conclusion (by rule):** {concl}")
    with open(f"{B.RES}/bang_tom_tat.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [10, 25, 50, 100]
    for row in ["S", "M-delta", "S+M", "T", "H-pred", "H-oracle"]:
        rr = [get(row, pn) for pn in PNAMES]
        if all(rr):
            ax.errorbar(xs, [r["bal_acc"] for r in rr],
                        yerr=[[r["bal_acc"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["bal_acc"] for r in rr]],
                        marker="o", capsize=3, label=row)
    ax.axhline(0.1, ls=":", c="gray", label="chance")
    ax.set_xlabel("% of video observed"); ax.set_ylabel("balanced accuracy"); ax.set_title("Part A: dish from prefix")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{B.RES}/curve_partA.png", dpi=130)
    if rowsB:
        fig, ax = plt.subplots(figsize=(8, 4))
        rows = ["SEG", "CTX-S", "HIST-pred", "HIST-oracle", "SEG+HIST-pred", "PRE"]
        acts = [c["action"] for c in astar] + ["MACRO"]
        wd = 0.8 / len(rows)
        for i, row in enumerate(rows):
            vals = [next((r["bal_acc"] for r in rowsB if r["row"] == row and r["action"] == a), np.nan) for a in acts]
            ax.bar(np.arange(len(acts)) + i * wd, vals, wd, label=row)
        ax.scatter(np.arange(len(acts)) + 0.4 - wd / 2, [1 / c["K"] for c in astar] + [np.mean([1 / c["K"] for c in astar])],
                   marker="_", s=300, c="k", label="chance", zorder=5)
        ax.set_xticks(np.arange(len(acts)) + 0.4 - wd / 2); ax.set_xticklabels(acts, rotation=20, fontsize=8)
        ax.set_ylabel("balanced accuracy"); ax.set_title("Part B: dish from one shared action")
        ax.legend(fontsize=7, ncol=3); fig.tight_layout(); fig.savefig(f"{B.RES}/partB_bar.png", dpi=130)
