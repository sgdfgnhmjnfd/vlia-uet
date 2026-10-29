# Stage 9d: egocentric video–language encoders, feasibility check only (2026-10-29)

Nothing was downloaded or installed. Sources:
- the GitHub READMEs, MODEL_ZOO and requirements/environment files of facebookresearch/LaViLa (archived Jul 2025), showlab/EgoVLP and facebookresearch/EgoVLPv2 (archived Aug 2025), read 2026-10-29;
- HTTP HEAD requests (`content-length` only) for the LaViLa checkpoint URLs;
- the EgoIntent dataset README at `/media/dhqg/d3/datasets/egointent_full/README.md`.

## Weights

| model | host | dual-encoder checkpoint | size |
|---|---|---|---|
| LaViLa TSF-B | dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/ | `clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth` | 711 MB |
| LaViLa TSF-L | same | `clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth` | 2.11 GB |
| EgoVLP, converted to the LaViLa codebase | same | `egovlp_converted_f16.md5sum_c33363.pth` | 724 MB |
| EgoVLP, original | Google Drive (`EgoVLP_PT_BEST`) | — | not stated |
| EgoVLPv2 | the README says checkpoints were released but gives no URL | — | not stated |

- LaViLa backbones are TimeSformer-B/L. The model zoo evaluates the dual encoder with 4 frames (EK-100 MIR, EgoMCQ) or 16 frames (Charades-Ego, EGTEA).
- EgoVLP is based on Frozen-in-Time and uses 4 frames zero-shot.
- Licences: LaViLa mostly MIT; EgoVLP MIT; EgoVLPv2 MIT.

## Packages

**lerobot env** (torch 2.11, transformers 5.5.4):
- has einops 0.8.2, av, pandas, regex;
- lacks timm, decord, ftfy, open_clip.

**Pinned requirements of each model:**
- **LaViLa:** torch 1.10.1, timm 0.5.4, decord 0.6.0, transformers 4.27, pytorchvideo 0.1.5, ftfy, spacy.
- **EgoVLP:** Python 3.8, torch 1.12.1+cu116, timm 0.4.12, transformers 4.2.1, decord 0.6.0.
- **EgoVLPv2:** torch 1.13.1+cu116, timm 0.4.12, transformers 4.30.0, decord, ftfy, ego4d.

None of them fits the lerobot env. Each would need a separate env on `/media/dhqg/d3`. It is not checked whether the old torch/CUDA 11.6 wheels run on this machine's GPU and driver.

## Disk

- `/`: 2.1 GB free, 100% used. Do not install there.
- `/media/dhqg/d3`: 1.5 TB free.

## Leakage risk: EgoIntent comes from Ego4D

- The EgoIntent README states that the benchmark is "constructed from 32 Ego4D source videos". Its `video_uid` field is the "Identifier of the Ego4D source video".
- LaViLa is pretrained on Ego4D narrations. EgoVLP and EgoVLPv2 are pretrained on EgoClip, which is built from Ego4D.
- None of the three READMEs says which Ego4D videos were excluded from pretraining.
- The EgoIntent videos, including the 4 locked-test videos, may therefore have been seen during pretraining, together with their narrations.

**This is a leakage risk for any result obtained with these encoders on EgoIntent.** Ruling it out would require checking the 32 EgoIntent `video_uid`s against the pretraining video lists (EgoClip metadata, LaViLa's Ego4D training split). That check was not done.
