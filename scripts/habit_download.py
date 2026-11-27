#!/usr/bin/env python3
"""Stage 22 Step 1: download parquet + front_view + human_front_view of the approved sids (Trang, STOP A 27/11:
S7414 + S8789). Resumable: snapshot_download skips finished files. Writes the file list with sizes to download_files.json."""
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
ROOT = Path("/media/dhqg/d3/datasets/habit")
SIDS = ("S7414", "S8789")
eps = [json.loads(l) for l in open(ROOT / "full/meta/episodes.jsonl")]
idx = sorted(e["episode_index"] for e in eps if e["sid"] in SIDS)
pats = []
for i in idx:
    c = f"chunk-{i//1000:03d}"
    pats += [f"full/data/{c}/episode_{i:06d}.parquet",
             f"full/videos/{c}/observation.images.front_view/episode_{i:06d}.mp4",
             f"full/videos/{c}/observation.images.human_front_view/episode_{i:06d}.mp4"]
print(len(idx), "episodes,", len(pats), "files", flush=True)
for k in range(0, len(pats), 150):
    snapshot_download(repo_id="configinc/HABIT", repo_type="dataset", allow_patterns=pats[k:k+150], local_dir=str(ROOT), max_workers=4)
    print(f"{min(k+150, len(pats))}/{len(pats)}", flush=True)
info = {p.path: p.size for k in range(0, len(pats), 300) for p in HfApi().get_paths_info("configinc/HABIT", pats[k:k+300], repo_type="dataset")}
bad = [p for p in pats if not (ROOT / p).exists() or (ROOT / p).stat().st_size != info.get(p)]
json.dump({"sids": SIDS, "episodes": idx, "files": info, "size_mismatch_or_missing": bad}, open(ROOT / "download_files.json", "w"), indent=0)
print("missing/mismatch:", len(bad), "total GB", sum(info.values()) / 1e9)
sys.exit(1 if bad else 0)
