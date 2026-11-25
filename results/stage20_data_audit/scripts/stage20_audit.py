"""Stage 20: data audit EgoIntent (read-only, CPU, no training / inference / accuracy re-run).

Sources (import, no checkout): /media/dhqg/d3/vlia-ego/scripts/stage18_scene_motion.py
(claude/stage18-scene-motion-ego 6702cf9; identical at HEAD claude/stage19-temporal 4942b23):
ego_data() (1,001 kept steps, KEEP, folds = run_scene_cv.assign_folds(scenes of all 2,503 steps, 8)),
verb(), lemma(), load_manifest() (asserts no locked-test video).
Writes only new files to /media/dhqg/d3/vlia-ego/results/stage20_data_audit/ (refuses to overwrite).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

EGO = Path("/media/dhqg/d3/vlia-ego")
sys.path.insert(0, str(EGO / "scripts"))
import stage18_scene_motion as S18  # noqa: E402
from run_scene_cv import assign_folds  # noqa: E402

OUT = EGO / "results/stage20_data_audit"
WORK = Path("/media/dhqg/d3/vlia_outputs/stage20_audit")
L = list(S18.KEEP)
K = len(L)


def new_file(name):
    p = OUT / name
    assert not p.exists(), f"refuse to overwrite {p}"
    return p


def wcsv(name, rows):
    with new_file(name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def class_verbs(t):
    ws = re.sub(r"[^a-z\s]", " ", t.lower()).split()
    found = []
    for i, w in enumerate(ws):
        lm = S18.lemma(w)
        v = "pick_up" if lm == "pick" and i + 1 < len(ws) and ws[i + 1] == "up" else (lm if lm in L else None)
        if v and v not in found:
            found.append(v)
    return found


def main():
    OUT.mkdir(parents=False, exist_ok=False)
    rep = {}
    # ---------------- A. data + split
    all_rows = S18.load_manifest()
    fm_all = assign_folds([r["scene"] for r in all_rows], S18.N_FOLDS_EGO)
    rows, _Z, y, vids, outer, dur = S18.ego_data()
    scenes = np.array([r["scene"] for r in rows])
    N = len(y)
    lab = {r["verb"]: int(r["steps"]) for r in csv.DictReader((S18.RES / "labels_ego.csv").open()) if r["kept"] == "1"}
    cnt = Counter(L[i] for i in y)
    checks = {
        "manifest_steps_2503": len(all_rows) == 2503,
        "manifest_videos": len({r["video_uid"] for r in all_rows}),
        "kept_steps_1001": N == 1001,
        "videos_27": len(set(vids)) == 27,
        "classes_10": K == 10,
        "folds_8": len(set(outer)) == 8,
        "class_counts_match_labels_ego_csv": lab == dict(cnt),
        "largest_class_0.255": round(max(cnt.values()) / N, 3),
        "match_stage18_oof_order": bool((np.load(S18.WORK / "ego_oof.npz")["y"] == y).all()
                                       and (np.load(S18.WORK / "ego_oof.npz")["outer"] == outer).all()),
        "no_locked_test_video": not (set(vids) & S18.locked_uids()),
        "each_video_one_scene": all(len({s for s, v in zip(scenes, vids) if v == u}) == 1 for u in set(vids)),
    }
    fold_tab = []
    for f in range(8):
        te, tr = outer == f, outer != f
        vo = set(vids[te]) & set(vids[tr])
        so = set(scenes[te]) & set(scenes[tr])
        fold_tab.append(dict(fold=f, val_scenes="|".join(sorted(set(scenes[te]))), train_videos=len(set(vids[tr])),
                             val_videos=len(set(vids[te])), train_steps=int(tr.sum()), val_steps=int(te.sum()),
                             video_overlap=len(vo), scene_overlap=len(so),
                             val_steps_all2503=sum(1 for r in all_rows if fm_all[r["scene"]] == f)))
    checks["no_video_overlap_any_fold"] = all(r["video_overlap"] == 0 for r in fold_tab)
    checks["no_scene_overlap_any_fold"] = all(r["scene_overlap"] == 0 for r in fold_tab)
    checks["val_steps_sum_1001"] = sum(r["val_steps"] for r in fold_tab) == N
    rep["checks"], rep["fold_tab"] = checks, fold_tab

    # ---------------- B. class x fold
    cfd = []
    for f in range(8):
        te, tr = outer == f, outer != f
        for c in range(K):
            mtr, mte = tr & (y == c), te & (y == c)
            vtr = Counter(vids[mtr].tolist())
            cfd.append(dict(fold=f, cls=L[c], train_steps=int(mtr.sum()), train_videos=len(vtr),
                            train_largest_video_share=float(max(vtr.values()) / mtr.sum()) if vtr else 0.0,
                            val_steps=int(mte.sum()), val_videos=len(set(vids[mte])),
                            train_class_share=float(mtr.sum() / tr.sum()), val_class_share=float(mte.sum() / te.sum()),
                            missing_in_train=int(mtr.sum() == 0), missing_in_val=int(mte.sum() == 0),
                            train_videos_le2=int(0 < len(vtr) <= 2),
                            val_share_of_class=float(mte.sum() / (y == c).sum())))
    wcsv("class_fold_distribution.csv", cfd)
    summ = []
    for c in range(K):
        rr = [r for r in cfd if r["cls"] == L[c]]
        vc = Counter(vids[y == c].tolist())
        summ.append(dict(cls=L[c], total_steps=int((y == c).sum()), total_videos=len(vc),
                         folds_missing_train=sum(r["missing_in_train"] for r in rr),
                         folds_missing_val=sum(r["missing_in_val"] for r in rr),
                         folds_missing_val_list="|".join(str(r["fold"]) for r in rr if r["missing_in_val"]),
                         min_train_steps=min(r["train_steps"] for r in rr), min_train_videos=min(r["train_videos"] for r in rr),
                         max_val_share_of_class=max(r["val_share_of_class"] for r in rr),
                         fold_of_max_val_share=max(rr, key=lambda r: r["val_share_of_class"])["fold"]))
    rep["class_summary"] = summ
    imb = []
    for f in range(8):
        te, tr = outer == f, outer != f
        ptr = np.bincount(y[tr], minlength=K) / tr.sum()
        pte = np.bincount(y[te], minlength=K) / te.sum()
        nz = pte[pte > 0]
        imb.append(dict(fold=f, train_max_share=float(ptr.max()), train_min_share=float(ptr.min()),
                        train_ratio_max_min=float(ptr.max() / ptr.min()), val_classes_present=int((pte > 0).sum()),
                        val_max_share=float(pte.max()), val_max_cls=L[pte.argmax()],
                        val_ratio_max_min_present=float(nz.max() / nz.min()),
                        tv_train_val=float(0.5 * np.abs(ptr - pte).sum())))
    rep["imbalance"] = imb

    # ---------------- C. scene / video
    S_names, V_names = sorted(set(scenes)), sorted(set(vids))
    svd = []
    for c in range(K):
        for s in S_names:
            for v in sorted(set(vids[scenes == s])):
                n = int(((vids == v) & (y == c)).sum())
                if n:
                    svd.append(dict(cls=L[c], scene=s, video=v, fold=int(outer[vids == v][0]), steps=n,
                                    share_of_class=float(n / (y == c).sum()),
                                    share_of_video=float(n / (vids == v).sum())))
    wcsv("scene_video_distribution.csv", svd)
    sv = []
    for c in range(K):
        sc = Counter(scenes[y == c].tolist())
        vc = Counter(vids[y == c].tolist())
        top_s, ns = sc.most_common(1)[0]
        n_vid_half = next(i + 1 for i, x in enumerate(np.cumsum(sorted(vc.values(), reverse=True)))
                          if x >= 0.5 * (y == c).sum())
        sv.append(dict(cls=L[c], scenes=len(sc), videos=len(vc), largest_video_share=float(max(vc.values()) / (y == c).sum()),
                       top_scene=top_s, top_scene_share=float(ns / (y == c).sum()), videos_for_half_steps=n_vid_half,
                       top_scene_videos=len(set(vids[(scenes == top_s)]))))
    rep["scene_video"] = sv
    rep["scenes"] = {s: dict(videos=len(set(vids[scenes == s])), steps=int((scenes == s).sum()),
                             fold=int(outer[scenes == s][0])) for s in S_names}

    # ---------------- D. labels
    whats = [r["what"] for r in rows]
    cv = [class_verbs(t) for t in whats]
    first_is_label = np.array([len(c) > 0 and c[0] == L[yy] for c, yy in zip(cv, y)])
    multi = np.array([len(c) >= 2 for c in cv])
    ws0 = [re.sub(r"[^a-z\s]", " ", t.lower()).split() for t in whats]
    gerund_first = np.array([w[0].endswith("ing") if w else False for w in ws0])
    pairs = [("place", "position"), ("pick_up", "grasp"), ("hold", "grasp"), ("adjust", "align"), ("remove", "place")]
    pair_rep = []
    rng = np.random.default_rng(0)
    for a, b in pairs:
        ia, ib = L.index(a), L.index(b)
        co = [i for i in range(N) if a in cv[i] and b in cv[i]]
        # object nouns after the verb (descriptive): top 5 words following the label verb
        def objs(c):
            o = Counter()
            for i in np.flatnonzero(y == c):
                w = [x for x in ws0[i] if x not in ("the", "a", "an", "his", "her", "their", "with", "on", "in", "into",
                                                     "to", "of", "up", "from", "left", "right", "hand", "both", "hands")]
                o.update(w[1:3])
            return [f"{k} {n}" for k, n in o.most_common(6)]
        sa = [whats[i] for i in sorted(rng.choice(np.flatnonzero(y == ia), 3, replace=False))]
        sb = [whats[i] for i in sorted(rng.choice(np.flatnonzero(y == ib), 3, replace=False))]
        sh_vid = len(set(vids[y == ia]) & set(vids[y == ib]))
        pair_rep.append(dict(pair=f"{a}/{b}", steps_a=int((y == ia).sum()), steps_b=int((y == ib).sum()),
                             same_sentence=len(co), example_same=[whats[i] for i in co[:2]],
                             shared_videos=sh_vid, words_a=objs(ia), words_b=objs(ib), ex_a=sa, ex_b=sb))
    rep["labels"] = dict(first_class_verb_is_label=float(first_is_label.mean()),
                         multi_class_verbs=float(multi.mean()), n_multi=int(multi.sum()),
                         gerund_first_word=float(gerund_first.mean()),
                         conj=float(np.mean([bool(re.search(r"\b(and|then|while)\b|,", t.lower())) for t in whats])),
                         second_verb_after_label=Counter(c[1] for c in cv if len(c) >= 2).most_common(10),
                         mean_words=float(np.mean([len(w) for w in ws0])), pairs=pair_rep)
    (WORK / "stage20_numbers.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
