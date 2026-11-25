"""Rà soát dữ liệu EgoIntent (25/11/2026): vì sao Stage 18/19 ra ngang may rủi trên cảnh chưa thấy.

Chỉ đọc + đếm, CPU, không train. Không sửa nhãn / tập nhãn / cách chia.

Nguồn code (import, không chép lại, không checkout):
  * /media/dhqg/d3/vlia-ego/scripts/stage18_scene_motion.py  (nhánh claude/stage18-scene-motion-ego 6702cf9;
    file giống hệt ở claude/stage19-temporal 4942b23, HEAD hiện tại của worktree)
      -> ego_data() (1.001 bước, 10 động từ, 8 fold theo cảnh qua run_scene_cv.assign_folds), verb(), lemma(),
         KEEP, feats(), bal_acc(), load_manifest() (assert không có video test khóa).
  * /media/dhqg/d3/vlia-ego/scripts/stage19_temporal.py (claude/stage19-temporal 4942b23) -> variant_X() cho VideoMAE.
Dự đoán đã lưu:
  * Stage 18: /media/dhqg/d3/vlia_outputs/stage18/ego_oof.npz (xác suất OOF từng bước, argmax = dự đoán)
  * Stage 19: /media/dhqg/d3/vlia_outputs/stage19/eval_ego.json ("pred" từng bước, cùng thứ tự ego_data()).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # không ghi __pycache__ vào worktree vlia-ego
import numpy as np

EGO = Path("/media/dhqg/d3/vlia-ego")
sys.path.insert(0, str(EGO / "scripts"))
import stage18_scene_motion as S18  # noqa: E402
import stage19_temporal as S19  # noqa: E402

OUT = Path("/media/dhqg/d3/vlia_outputs/ra_soat_ego")
RES = OUT / "results"
K = len(S18.KEEP)
L = list(S18.KEEP)
FPS_FALLBACK = 30.0


def wcsv(name, rows, header=None):
    header = header or list(rows[0].keys())
    with (RES / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def norm_entropy(counts):
    c = np.asarray([x for x in counts if x > 0], float)
    if len(c) <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    rows, Z, y, vids, outer, dur = S18.ego_data()
    scenes = np.array([r["scene"] for r in rows])
    N = len(y)
    # ---- kiểm đầu vào
    lab = list(csv.DictReader((S18.RES / "labels_ego.csv").open()))
    lab_cnt = {r["verb"]: int(r["steps"]) for r in lab if r["kept"] == "1"}
    cnt = Counter(L[i] for i in y)
    assert N == 1001 and K == 10 and len(set(vids)) == 27 and len(set(outer)) == 8, (N, K, len(set(vids)))
    assert lab_cnt == dict(cnt), (lab_cnt, cnt)
    assert abs(max(cnt.values()) / N - 0.2547) < 1e-3
    assert not (set(vids) & S18.locked_uids())
    for f in range(8):
        assert not (set(vids[outer == f]) & set(vids[outer != f]))
    oof = np.load(S18.WORK / "ego_oof.npz")
    assert (oof["y"] == y).all() and (oof["vids"] == vids).all() and (oof["outer"] == outer).all()
    print(f"OK: {N} bước, {K} lớp, {len(set(vids))} video, {len(set(scenes))} cảnh, 8 fold; lớn nhất {max(cnt.values())/N:.3f}")
    rep = {}

    # ---- nhóm cảnh
    sc_vids = defaultdict(set)
    for s, v in zip(scenes, vids):
        sc_vids[s].add(v)
    fold_sc = defaultdict(set)
    for s, f in zip(scenes, outer):
        fold_sc[int(f)].add(s)
    rep["scenes"] = {s: len(v) for s, v in sorted(sc_vids.items())}
    rep["fold_scenes"] = {f: sorted(s) for f, s in sorted(fold_sc.items())}
    vid_sc = defaultdict(set)
    for s, v in zip(scenes, vids):
        vid_sc[v].add(s)
    assert all(len(s) == 1 for s in vid_sc.values())

    # ================= 1. fold × lớp (H-a)
    fl, fold_sum = [], []
    n_pairs = n_flag = n_zero = n_fewvid = n_fewtest = 0
    for f in range(8):
        te, tr = outer == f, outer != f
        ptr = np.bincount(y[tr], minlength=K) / tr.sum()
        pte = np.bincount(y[te], minlength=K) / te.sum()
        for c in range(K):
            ntr, nte = int((y[tr] == c).sum()), int((y[te] == c).sum())
            vtr, vte = len(set(vids[tr][y[tr] == c])), len(set(vids[te][y[te] == c]))
            z0 = nte > 0 and ntr == 0
            fv = nte > 0 and vtr < 3
            ft = 0 < nte < 5
            if nte > 0:
                n_pairs += 1
                n_zero += z0
                n_fewvid += fv
                n_fewtest += ft
            fl.append(dict(fold=f, lop=L[c], buoc_train=ntr, buoc_test=nte, video_train=vtr, video_test=vte,
                           test_nhung_train0=int(z0), video_train_lt3=int(fv), test_lt5=int(ft)))
        fold_sum.append(dict(fold=f, canh_test="|".join(sorted(fold_sc[f])), buoc_test=int(te.sum()),
                             video_test=len(set(vids[te])), so_lop_test=int((pte > 0).sum()),
                             lop_nhieu_nhat_train=L[ptr.argmax()], ti_le_train=float(ptr.max()),
                             lop_nhieu_nhat_test=L[pte.argmax()], ti_le_test=float(pte.max()),
                             tv_train_test=float(0.5 * np.abs(ptr - pte).sum())))
    wcsv("fold_lop.csv", fl)
    wcsv("fold_tom_tat.csv", fold_sum)
    rep["Ha"] = dict(cap_co_test=n_pairs, train0=n_zero, video_train_lt3=n_fewvid, test_lt5=n_fewtest,
                     tv_mean=float(np.mean([r["tv_train_test"] for r in fold_sum])),
                     tv_max=float(np.max([r["tv_train_test"] for r in fold_sum])))

    # ================= 2. cảnh / video × lớp (H-b)
    S_names = sorted(set(scenes))
    V_names = sorted(set(vids))
    cs = [dict(canh=s, so_video=len(sc_vids[s]), fold=int(outer[scenes == s][0]), tong=int((scenes == s).sum()),
               **{L[c]: int(((scenes == s) & (y == c)).sum()) for c in range(K)}) for s in S_names]
    wcsv("canh_lop.csv", cs)
    vl = []
    for v in V_names:
        m = vids == v
        bc = np.bincount(y[m], minlength=K)
        vl.append(dict(video=v, canh=scenes[m][0], tong=int(m.sum()), so_dong_tu=int((bc > 0).sum()),
                       dong_tu_nhieu_nhat=L[bc.argmax()], ti_le=float(bc.max() / m.sum()),
                       **{L[c]: int(bc[c]) for c in range(K)}))
    wcsv("video_lop.csv", vl)
    hb = []
    for c in range(K):
        sc = np.array([((scenes == s) & (y == c)).sum() for s in S_names])
        vc = np.array([((vids == v) & (y == c)).sum() for v in V_names])
        srt = np.sort(sc)[::-1]
        vsrt = np.sort(vc)[::-1]
        hb.append(dict(lop=L[c], buoc=int(sc.sum()), so_canh=int((sc > 0).sum()), so_video=int((vc > 0).sum()),
                       canh_top=S_names[sc.argmax()], ti_le_canh_top1=float(srt[0] / sc.sum()),
                       ti_le_canh_top2=float(srt[:2].sum() / sc.sum()), entropy_canh=norm_entropy(sc),
                       ti_le_video_top1=float(vsrt[0] / vc.sum()), ti_le_video_top2=float(vsrt[:2].sum() / vc.sum()),
                       entropy_video=norm_entropy(vc)))
    wcsv("lop_tap_trung.csv", hb)
    rep["Hb"] = dict(lop_gt50_1canh=[r["lop"] for r in hb if r["ti_le_canh_top1"] > 0.5],
                     lop_gt50_1video=[r["lop"] for r in hb if r["ti_le_video_top1"] > 0.5])

    # ================= 3. nhãn (H-c)
    whats = [r["what"] for r in rows]

    def class_verbs(t):
        ws = re.sub(r"[^a-z\s]", " ", t.lower()).split()
        found = set()
        for i, w in enumerate(ws):
            lm = S18.lemma(w)
            if lm == "pick" and i + 1 < len(ws) and ws[i + 1] == "up":
                found.add("pick_up")
            elif lm in L:
                found.add(lm)
        return found

    multi = np.array([len(class_verbs(t)) >= 2 for t in whats])
    conj = np.array([bool(re.search(r"\b(and|then)\b|,", t.lower())) for t in whats])
    rep["Hc"] = dict(ti_le_2dongtu=float(multi.mean()), ti_le_and_then_phay=float(conj.mean()))
    rng = np.random.default_rng(0)
    mau = []
    for c in range(K):
        idx = np.flatnonzero(y == c)
        for i in sorted(rng.choice(idx, size=min(5, len(idx)), replace=False)):
            mau.append(dict(lop=L[c], what=whats[i], video=vids[i][:8], canh=scenes[i], do_dai_s=float(dur[i]),
                            dong_tu_lop_trong_cau="|".join(sorted(class_verbs(whats[i])))))
    wcsv("mau_cau_what.csv", mau)
    pair_multi = Counter()
    for t in whats:
        cv = sorted(class_verbs(t))
        for a in range(len(cv)):
            for b in range(a + 1, len(cv)):
                pair_multi[(cv[a], cv[b])] += 1
    rep["Hc"]["cap_dong_tu_cung_cau"] = [f"{a}+{b}: {n}" for (a, b), n in pair_multi.most_common(6)]

    # ================= 4. recall / phân bố dự đoán (H-f)
    E19 = json.loads((S19.WORK / "eval_ego.json").read_text())
    preds = {f"S18 {k}": oof[k].argmax(1) for k in ("S|0", "M|50", "S+M|50", "S+M|100")}
    preds.update({f"S19 {k}": np.array(E19[k]["pred"]) for k in ("S|0", "M|50", "S+M|50", "S+M|100")})
    near = [{"place", "position", "align"}, {"grasp", "pick_up", "hold"}, {"adjust", "align"}]
    rc, pdist, hf, conf_pairs = [], [], {}, []
    for name, p in preds.items():
        ba = S18.bal_acc(y, p, K)
        rec = np.array([(p[y == c] == c).mean() for c in range(K)])
        pc = np.bincount(p, minlength=K)
        err = p != y
        nearerr = sum(1 for a, b in zip(y[err], p[err]) if any({L[a], L[b]} <= g for g in near))
        top2 = np.sort(pc)[::-1][:2].sum() / N
        hf[name] = dict(bal_acc=float(ba), acc=float((p == y).mean()), recall0=int((rec == 0).sum()),
                        top2_du_doan=float(top2), loi_gan_nghia=float(nearerr / err.sum()),
                        lop_du_doan_nhieu=[f"{L[i]} {pc[i]/N:.2f}" for i in np.argsort(-pc)[:3]])
        for c in range(K):
            rc.append(dict(cau_hinh=name, lop=L[c], n_that=int((y == c).sum()), recall=float(rec[c]),
                           so_du_doan=int(pc[c]), ti_le_du_doan=float(pc[c] / N)))
        cm = np.zeros((K, K), int)
        np.add.at(cm, (y, p), 1)
        off = [(cm[a, b], L[a], L[b]) for a in range(K) for b in range(K) if a != b]
        for n_, a, b in sorted(off, reverse=True)[:5]:
            conf_pairs.append(dict(cau_hinh=name, that=a, doan=b, so_buoc=int(n_)))
    wcsv("recall_lop.csv", rc)
    wcsv("cap_nham_nhieu.csv", conf_pairs)
    wcsv("phan_bo_du_doan.csv", [dict(cau_hinh=k, **{kk: (vv if not isinstance(vv, list) else "; ".join(vv))
                                                       for kk, vv in v.items()}) for k, v in hf.items()])
    rep["Hf"] = hf

    # ================= 5. độ dài, cửa sổ (H-d)
    import cv2
    nfr = np.zeros(N, int)
    fps = np.zeros(N)
    for i, r in enumerate(rows):
        cap = cv2.VideoCapture(r["video_path"])
        nfr[i], fps[i] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        cap.release()

    def idx8(n):  # như stage19_temporal.ego_end_frame (khớp cache Stage 18)
        a = np.linspace(0, n - 1, num=min(8, n), dtype=int)
        if len(a) < 8:
            a = a[np.linspace(0, len(a) - 1, 8, dtype=np.float32).round().astype(int)]
        return a

    I8 = np.stack([idx8(max(n, 1)) for n in nfr])
    win = {p: I8[:, kf - 1] / fps for p, kf in S18.EGO_P.items()}
    gap = (I8[:, 1] - I8[:, 0]) / fps
    dup50 = np.array([len(set(a[:4])) < 4 for a in I8])  # p50 có khung lặp
    m50 = np.linalg.norm(np.stack([S18.feats(z, 4)["M"] for z in Z]), axis=1)
    rec_sm50 = {k: np.array([(preds[k][y == c] == c).mean() for c in range(K)]) for k in ("S18 S+M|50", "S19 S+M|50")}
    dl = []
    for c in range(K):
        m = y == c
        d = dur[m]
        dl.append(dict(lop=L[c], n=int(m.sum()), tb_s=float(d.mean()), trung_vi_s=float(np.median(d)),
                       p10_s=float(np.percentile(d, 10)), p90_s=float(np.percentile(d, 90)),
                       cua_so_p25_tv_s=float(np.median(win[25][m])), cua_so_p50_tv_s=float(np.median(win[50][m])),
                       cua_so_p100_tv_s=float(np.median(win[100][m])), khoang_khung_tv_s=float(np.median(gap[m])),
                       p50_lt_05s=int((win[50][m] < 0.5).sum()), p50_lt_1s=int((win[50][m] < 1).sum()),
                       p50_khung_lap=int(dup50[m].sum()),
                       recall_S18_SM50=float(rec_sm50["S18 S+M|50"][c]), recall_S19_SM50=float(rec_sm50["S19 S+M|50"][c])))
    wcsv("do_dai_lop.csv", dl)
    from scipy.stats import spearmanr
    med = [r["trung_vi_s"] for r in dl]
    rep["Hd"] = dict(trung_vi_buoc_s=float(np.median(dur)), tb_buoc_s=float(dur.mean()),
                     trung_vi_p25_s=float(np.median(win[25])), trung_vi_p50_s=float(np.median(win[50])),
                     trung_vi_p100_s=float(np.median(win[100])), trung_vi_khoang_khung_s=float(np.median(gap)),
                     p50_lt_05s=int((win[50] < 0.5).sum()), p50_lt_1s=int((win[50] < 1).sum()),
                     p50_khung_lap=int(dup50.sum()), fps=sorted(set(np.round(fps, 2).tolist())),
                     M50_norm_trung_vi=float(np.median(m50)),
                     spearman_dodai_recall_S18=[float(x) for x in spearmanr(med, rec_sm50["S18 S+M|50"])],
                     spearman_dodai_recall_S19=[float(x) for x in spearmanr(med, rec_sm50["S19 S+M|50"])])

    # ================= 6. láng giềng (H-e), không train
    def l2(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    S_ = np.stack([z[0] for z in Z])
    M_ = np.stack([S18.feats(z, 4)["M"] for z in Z])
    keys = [f"{i:04d}" for i in range(N)]
    spaces = {"SigLIP S": l2(S_), "SigLIP S+M p50 (L2 từng khối rồi ghép)": l2(np.hstack([l2(S_), l2(M_)])),
              "SigLIP M p50": l2(M_)}
    try:
        spaces["VideoMAE S+M p50 (= vector clip V50)"] = l2(S19.variant_X("ego", keys, "S+M", 50))
        spaces["VideoMAE S"] = l2(S19.variant_X("ego", keys, "S", 0))
    except Exception as e:  # noqa: BLE001
        rep["He_videomae_err"] = repr(e)
    same_v = vids[:, None] == vids[None, :]
    same_s = scenes[:, None] == scenes[None, :]
    same_y = y[:, None] == y[None, :]
    eye = np.eye(N, dtype=bool)
    exp_v = ((same_v & ~eye).sum(1) / (N - 1)).mean()
    exp_s = ((same_s & ~eye).sum(1) / (N - 1)).mean()
    exp_y = ((same_y & ~eye).sum(1) / (N - 1)).mean()
    exp_y_other = ((same_y & ~same_s).sum(1) / (~same_s).sum(1)).mean()
    exp_y_samevid = np.nanmean(np.where((same_v & ~eye).sum(1) > 0,
                                        (same_y & same_v & ~eye).sum(1) / np.maximum((same_v & ~eye).sum(1), 1), np.nan))
    lg = []
    for name, X in spaces.items():
        sim = X @ X.T
        sim[eye] = -np.inf
        nn = sim.argmax(1)
        sim_o = np.where(same_s, -np.inf, sim)
        nno = sim_o.argmax(1)
        r = dict(dac_trung=name, cung_video=float((vids[nn] == vids).mean()), ky_vong_cung_video=float(exp_v),
                 cung_canh=float((scenes[nn] == scenes).mean()), ky_vong_cung_canh=float(exp_s),
                 cung_dong_tu=float((y[nn] == y).mean()), ky_vong_cung_dong_tu=float(exp_y),
                 cung_dong_tu_canh_khac=float((y[nno] == y).mean()), ky_vong_canh_khac=float(exp_y_other))
        r["boi_cung_video"] = r["cung_video"] / exp_v
        r["boi_dong_tu_canh_khac"] = r["cung_dong_tu_canh_khac"] / exp_y_other
        lg.append(r)
    wcsv("lang_gieng.csv", lg)
    rep["He"] = dict(rows=lg, ky_vong_cung_dong_tu_trong_cung_video=float(exp_y_samevid))

    rep["giay"] = time.time() - t0
    (RES / "ra_soat.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
