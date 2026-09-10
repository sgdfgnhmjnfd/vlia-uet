from pathlib import Path

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
)


REPO_ID = "py2279105943/EgoIntent"

ROOT = Path(
    "/media/dhqg/d1/datasets/egointent"
)


EVENTS = [
    "indoor/bedroom/sort_clothes",
    "indoor/garage/repair_bicycle",
    "indoor/kitchen/boil_noodles",
    "indoor/kitchen/wash_dishes",
    "indoor/workshop/assemble_wood",
    "outdoor/farm/pick_fruits",
    "outdoor/garden/cut_branches",
]


repo_files = list_repo_files(
    REPO_ID,
    repo_type="dataset",
)


selected = []

for filename in repo_files:
    if not any(
        filename.startswith(event + "/")
        for event in EVENTS
    ):
        continue

    if (
        filename.endswith(".mp4")
        or filename.endswith("step_label.json")
    ):
        selected.append(filename)


print("files to download:", len(selected))


for index, filename in enumerate(
    selected,
    start=1,
):
    print(
        f"[{index}/{len(selected)}] "
        f"{filename}"
    )

    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=ROOT,
    )


print()
print("EGOINTENT PILOT DOWNLOAD: DONE")