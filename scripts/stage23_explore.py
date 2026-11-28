"""Stage 23 EXPLORATORY (not pre-registered): causal dilated TCN for fine actions + dish.

E1  F-TCN: causal dilated TCN on 1-s PCA-256 sequences -> per-second fine label (48 cls). Gives H-pred2 / HIST-pred2.
E2  T2   : same TCN, multi-task (fine label + dish per second, causal). Dish at prefix n = mean softmax over observed secs.
Train people only; train-video features for the linear stage are cross-fitted (3 person groups).
PCA = the split's full-train PCA from the main run (unsupervised; fit on train people only).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage23_breakfast as B  # noqa: E402
import stage23_core as C  # noqa: E402

EXP = os.path.join(B.RES, "explore")
FOLDS = os.path.join(EXP, "folds")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tcn(K_out_fine=48, K_dish=10, ch=64, layers=10):
    import torch.nn as nn
    import torch

    class Layer(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.d = d
            s.conv = nn.Conv1d(ch, ch, 3, dilation=d)
            s.out = nn.Conv1d(ch, ch, 1)
            s.drop = nn.Dropout(0.3)

        def forward(s, x):
            h = nn.functional.pad(x, (2 * s.d, 0))  # causal
            return x + s.drop(s.out(torch.relu(s.conv(h))))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Conv1d(256, ch, 1)
            s.layers = nn.ModuleList([Layer(2 ** i) for i in range(layers)])
            s.fine = nn.Conv1d(ch, K_out_fine, 1)
            s.dish = nn.Conv1d(ch, K_dish, 1)

        def forward(s, x):
            h = s.inp(x)
            for l in s.layers:
                h = l(h)
            return s.fine(h), s.dish(h)

    return Net()


def train_tcn(Z, L, y_dish, vids, seed, epochs=25, use_dish=True):
    import torch
    import torch.nn as nn
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = tcn()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    order = np.array(vids)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(order))
        for i in range(0, len(perm), 8):
            ids = order[perm[i:i + 8]]
            Lm = max(Z[v].shape[0] for v in ids)
            x = torch.zeros(len(ids), 256, Lm)
            yf = torch.full((len(ids), Lm), -1, dtype=torch.long)
            yd = torch.full((len(ids), Lm), -1, dtype=torch.long)
            for j, v in enumerate(ids):
                n = Z[v].shape[0]
                x[j, :, :n] = torch.from_numpy(Z[v].T)
                yf[j, :n] = torch.from_numpy(L[v])
                yd[j, :n] = int(y_dish[v])
            of, od = net(x)
            loss = ce(of, yf)
            if use_dish:
                loss = loss + ce(od, yd)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


def infer(net, Z, vids):
    import torch
    out = {}
    with torch.no_grad():
        for v in vids:
            of, od = net(torch.from_numpy(Z[v].T)[None])
            out[v] = (of[0].argmax(0).numpy().astype(np.int64),
                      torch.softmax(od[0], 0).numpy().T.astype(np.float32))  # (K,10)
    return out


def save(name, obj):
    os.makedirs(FOLDS, exist_ok=True)
    with open(f"{FOLDS}/{name}.json.tmp", "w") as f:
        json.dump(obj, f)
    os.replace(f"{FOLDS}/{name}.json.tmp", f"{FOLDS}/{name}.json")


def have(name):
    return os.path.exists(f"{FOLDS}/{name}.json")


def main():
    vids, m, dishes, people = C.meta_all()
    dix = {d: i for i, d in enumerate(dishes)}
    names, idx = C.mapping()
    astar = json.load(open(f"{B.RES}/astar.json"))
    sp = B.read_splits()
    splits = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4").split(",")]
    for k in splits:
        t0 = time.time()
        tr, te = sp[k]["train"], sp[k]["test"]
        zd = np.load(f"{B.CACHE}/pca_split{k}.npz", allow_pickle=True)
        Z = zd["Z"].item()
        L = {}
        for v in tr + te:
            with C.load_npz(v) as d:
                L[v] = d["lab1s"].astype(np.int64)
        yd = {v: dix[m[v][2]] for v in tr + te}
        pp = f"{B.CACHE}/explore_pred_split{k}.npz"
        if os.path.exists(pp):
            P = np.load(pp, allow_pickle=True)["P"].item()
        else:
            P = {}
            # test: 3 seeds full train; fine labels from seed 0, dish probs averaged over seeds
            for seed in (0, 1, 2):
                net = train_tcn(Z, L, yd, tr, seed)
                o = infer(net, Z, te)
                for v in te:
                    if seed == 0:
                        P[v] = dict(fine=o[v][0], dish=[o[v][1]])
                    else:
                        P[v]["dish"].append(o[v][1])
                log(f"split{k} tcn seed{seed} done {time.time()-t0:.0f}s")
            # cross-fit fine labels for train videos (seed 0)
            trp = sorted({m[v][0] for v in tr})
            grp = {p: i % 3 for i, p in enumerate(trp)}
            for g in range(3):
                fit_v = [v for v in tr if grp[m[v][0]] != g]
                net = train_tcn(Z, L, yd, fit_v, 0)
                o = infer(net, Z, [v for v in tr if grp[m[v][0]] == g])
                for v, (f, dsh) in o.items():
                    P[v] = dict(fine=f, dish=[dsh])
                log(f"split{k} crossfit g{g} done {time.time()-t0:.0f}s")
            np.savez(pp + ".tmp.npz", P=np.array(P, dtype=object))
            os.replace(pp + ".tmp.npz", pp)
        acc_raw = float(np.mean([(P[v]["fine"] == L[v]).mean() for v in te]))
        save(f"frameclf2_split{k}", dict(test_sec_acc_raw=acc_raw))
        log(f"split{k} TCN fine sec-acc test {acc_raw:.3f}")
        sm = {v: C.smooth_causal(P[v]["fine"]) for v in tr + te}
        D = {v: dict(C.load_npz(v)) for v in tr + te}
        ytr = np.array([yd[v] for v in tr]); yte = np.array([yd[v] for v in te])
        gtr = np.array([m[v][0] for v in tr])
        # E2: T2 dish from TCN head, per seed and averaged
        for pi, pn in enumerate(C.PNAMES):
            for seed in (0, 1, 2):
                name = f"A_T2_0_{pn}_split{k}_seed{seed}"
                if not have(name):
                    yhat = [int(P[v]["dish"][seed][: max(1, int(D[v]["n0"][pi]) // B.FPS)].mean(0).argmax()) for v in te]
                    save(name, dict(vids=te, y=yte.tolist(), yhat=yhat, info={}, majority=0))
            name = f"A_T2ens_0_{pn}_split{k}"
            if not have(name):
                yhat = [int(np.mean([q[: max(1, int(D[v]["n0"][pi]) // B.FPS)].mean(0) for q in P[v]["dish"]], 0).argmax()) for v in te]
                save(name, dict(vids=te, y=yte.tolist(), yhat=yhat, info={}, majority=0))
                log(name, f"bal {C.metrics_point(yte, np.array(yhat), 10)['bal_acc']:.3f}")
            # E1: H-pred2 rows
            for row in ("H-pred2", "S+M+H-pred2"):
                name = f"A_{row}_0_{pn}_split{k}"
                if have(name):
                    continue
                def f(v):
                    h = C.hist_pred_upto(sm[v], int(D[v]["n0"][pi]))
                    if row == "H-pred2":
                        return h
                    return np.concatenate([D[v]["S0"], D[v]["pm0"][pi] - D[v]["x0"], h])
                Xtr = np.stack([f(v) for v in tr]); Xte = np.stack([f(v) for v in te])
                yhat, info = C.fit_linear(Xtr, ytr, gtr, Xte)
                save(name, dict(vids=te, y=yte.tolist(), yhat=yhat.tolist(), info=info, majority=0))
                log(name, f"bal {C.metrics_point(yte, yhat, 10)['bal_acc']:.3f}")
        # Part B with HIST-pred2
        itr = C.segB_index(tr, m, astar, idx); ite = C.segB_index(te, m, astar, idx)
        for c in astar:
            a = c["action"]; cix = {d: i for i, d in enumerate(c["dishes"])}
            Rtr = [r for r in itr if r[0] == a]; Rte = [r for r in ite if r[0] == a]
            ytrb = np.array([cix[r[3]] for r in Rtr]); yteb = np.array([cix[r[3]] for r in Rte])
            gtrb = np.array([m[r[1]][0] for r in Rtr])
            for row in ("HIST-pred2", "SEG+HIST-pred2"):
                name = f"B_{row}_{a}_split{k}"
                if have(name):
                    continue
                if len(Rte) == 0:
                    save(name, dict(vids=[], segj=[], y=[], yhat=[], info={"empty_test": True}, K=len(cix), majority=0))
                    continue
                def g(r):
                    d = D[r[1]]; s = int(d["segs"][r[2], 1])
                    h = C.hist_pred_upto(sm[r[1]], s)
                    return h if row == "HIST-pred2" else np.concatenate([d["seg_mean"][r[2]].astype(np.float32), h])
                Xtr = np.stack([g(r) for r in Rtr]); Xte = np.stack([g(r) for r in Rte])
                yhat, info = C.fit_linear(Xtr, ytrb, gtrb, Xte)
                save(name, dict(vids=[r[1] for r in Rte], segj=[r[2] for r in Rte], y=yteb.tolist(),
                                yhat=yhat.tolist(), info=info, K=len(cix), majority=0))
                log(name, f"bal {C.metrics_point(yteb, yhat, len(cix))['bal_acc']:.3f}")
        log(f"split{k} explore done {time.time()-t0:.0f}s")
        del Z, D, P, sm


if __name__ == "__main__":
    main()
