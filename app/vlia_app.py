#!/usr/bin/env python3
"""VLIA demo app: video of a person -> predicted intention -> robot in LIBERO.

Tab 1 (intention): upload a video, the app shows the intention predicted by the three modules
after 25 / 50 / 75 / 100 % of the action:
  C  supervised classifier (needs app/classifier_svm_lin.npz from scripts/train_app_classifier.py)
  F  training-free sentence matching with SigLIP, plus "ask the person" when the margin is small
  G  Qwen2.5-VL writes the intention itself, without a label list
Tab 2 (robot): the chosen intention sentence becomes the intention token of VLIA, which runs one
episode in a LIBERO-10 scene with an empty instruction; the app shows the rollout video.

Run on the GPU machine (see README):
  python app/vlia_app.py --vlia-checkpoint <.../pretrained_model> [--share]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage12_e2e as S12  # noqa: E402
import run_stage13_human as H  # noqa: E402
import run_stage15_supervised as S15  # noqa: E402

PCTS = (25, 50, 75, 100)
TAU = 0.0106  # ask threshold of module F, fixed on calibration demos (thesis, closed-loop experiment)


def task_descriptions() -> dict[int, str]:
    d = json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"]
    return {int(k): v for k, v in d.items()}


class Models:
    """Loaded lazily: SigLIP for C and F, Qwen only when G is used."""

    def __init__(self, classifier: str | None, context_mean: str | None):
        import torch

        import extract_ladder_embeddings as X

        self.X = X
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siglip, self.proc = X.load_encoder(S12.ENCODER, self.device)
        self.tasks = task_descriptions()
        texts = [S12.TEMPLATES["T1"].format(why=self.tasks[t]) for t in range(len(self.tasks))]
        emb, _ = X.text_features(self.siglip, self.proc, S12.ENCODER, texts, self.device, 32)
        self.task_text = S12.normalize(np.asarray(emb, dtype=np.float64))
        self.clf = np.load(classifier) if classifier and Path(classifier).exists() else None
        self.mu = None
        if context_mean and Path(context_mean).exists():
            z = np.load(context_mean)
            self.mu = z["context_mean"] if "context_mean" in z.files else z[z.files[0]]
        self._qwen = None

    def image_emb(self, frames: np.ndarray) -> np.ndarray:
        return S12.encode(self.siglip, self.proc, frames, self.device, self.X).astype(np.float64)

    def predict_c(self, seq: np.ndarray):
        if self.clf is None:
            return None
        s = S15.mean_feat(seq[None])[0] @ self.clf["W"] + self.clf["b"]
        t = int(np.argmax(s))
        return t, self.tasks[t]

    def predict_f(self, seq: np.ndarray):
        q = S12.query_vec(seq, self.mu)
        s = self.task_text @ q
        order = np.argsort(-s)
        margin = float(s[order[0]] - s[order[1]])
        t = int(order[0])
        return t, self.tasks[t], margin, margin < TAU

    def predict_g(self, frames: np.ndarray) -> str:
        import run_stage14_generate as G

        if self._qwen is None:
            self._qwen = G.Generator()
        import v3_stage3_mllm as M3

        sent, _raw, _n = self._qwen([M3.resize448(f) for f in frames], G.PROMPT_LIB)
        return sent


def read_video(path: str, start_s: float | None, end_s: float | None):
    """4 fps frames of the whole clip, and the action window [start, end] in seconds."""
    m = H.probe(Path(path))
    idx, times = S15.frame_grid(m["n_frames"], m["fps"])
    fr, _, _ = H.decode(Path(path), wanted={int(i) for i in idx})
    frames = np.stack([fr[int(i)] for i in idx])
    start = 0.0 if start_s is None or start_s < 0 else float(start_s)
    end = float(times[-1]) if end_s is None or end_s <= start else min(float(end_s), float(times[-1]))
    return frames, times, start, end


TITLE = "Tăng cường khả năng hiểu ý định người dùng cho robot cộng tác dựa trên mô hình VLA"
AUTHOR = "Vũ Thị Thu Trang · Kỹ thuật Robot · Trường Đại học Công nghệ, ĐHQGHN"
ADVISOR = "Cán bộ hướng dẫn: PGS. TS. Hoàng Văn Xiêm"

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto; font-family: 'Be Vietnam Pro', 'Inter', sans-serif;}
#hero {background: linear-gradient(120deg, #0b3d91 0%, #1565c0 55%, #26a0da 100%); border-radius: 20px;
       padding: 14px 22px; color: #fff; box-shadow: 0 10px 30px rgba(11,61,145,.25);}
#hero .row {display: flex; align-items: center; gap: 18px;}
#hero img {width: 66px; height: 66px; border-radius: 50%; background: #fff; padding: 4px;}
#hero .tag {display: inline-block; font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
            background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px;}
#hero h1 {color: #fff; font-size: 22px; line-height: 1.25; margin: 4px 0 4px;}
#hero p {color: rgba(255,255,255,.88); margin: 0; font-size: 13px;}
#flow {display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 0 0;}
#flow span.s {background: #eef4ff; color: #0b3d91; border: 1px solid #cfe0ff; border-radius: 999px;
              padding: 3px 11px; font-size: 12px; font-weight: 600;}
#flow span.a {color: #7a8bb0;}
.card {border-radius: 16px !important; box-shadow: 0 4px 18px rgba(16,40,90,.08) !important; border: 1px solid #e6ecf7 !important;}
.primary-btn {background: linear-gradient(90deg, #1565c0, #26a0da) !important; color: #fff !important; border: none !important;
              border-radius: 12px !important; font-weight: 700 !important; font-size: 16px !important;
              box-shadow: 0 6px 16px rgba(21,101,192,.35) !important; transition: transform .1s;}
.primary-btn:hover {transform: translateY(-1px);}
table.res {width: 100%; border-collapse: separate !important; border-spacing: 0 5px; font-size: 14px; border: none !important; margin: 0 !important;}
table.res th, table.res td, table.res tr, table.res thead, table.res tbody {border: none !important;}
table.res tr {background: transparent !important;}
body footer, .gradio-container > footer {display: none !important;}
table.res th {text-align: left; color: #5b6b8c; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; padding: 0 12px; white-space: nowrap; background: transparent !important;}
table.res td {background: #f2f6fc !important; padding: 8px 12px !important; vertical-align: middle; line-height: 1.35;}
table.res td b {color: #13213d; font-weight: 600;}
table.res td:first-child {border-radius: 10px 0 0 10px; font-weight: 800; font-size: 16px; color: #0b3d91; white-space: nowrap; width: 64px;}
table.res td:last-child {border-radius: 0 10px 10px 0;}
.badge {display: inline-block; font-size: 11px; font-weight: 700; border-radius: 6px; padding: 1px 7px; margin-right: 6px; color: #fff;}
.bC {background: #2e7d32;} .bF {background: #1565c0;} .bG {background: #8e24aa;}
.ask {display: inline-block; margin-left: 6px; font-weight: 600; font-size: 12px; color: #b26a00; background: #fff4e0; border-radius: 6px; padding: 1px 7px;}
.muted {color: #8a94a8;}
#foot {text-align: center; color: #8a94a8; font-size: 12px; margin-top: 4px;}
.nbox {background: #eef4ff; border-left: 4px solid #1565c0; border-radius: 10px; padding: 6px 12px !important; font-size: 13px;}
.nbox b {color: #0b3d91;}
.nbox pre {white-space: pre-wrap; font-size: 12px; margin: 4px 0 0;}
.hint {font-size: 12px; color: #5b6b8c; margin: -2px 0 0;}
.hint p {margin: 0 !important;}
.card {padding: 10px !important;}
"""


def _logo_b64() -> str:
    import base64

    p = ROOT / "app/assets/uet_logo.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def hero_html() -> str:
    logo = _logo_b64()
    img = f'<img src="data:image/png;base64,{logo}" alt="UET">' if logo else ""
    flow = "".join(f'<span class="s">{x}</span><span class="a">→</span>' for x in
                   ("Video người đang làm", "Đoán ý định", "Câu ý định", "Token ý định")) + '<span class="s">Robot hành động</span>'
    return (f'<div id="hero"><div class="row">{img}<div><span class="tag">Đồ án tốt nghiệp · Demo VLIA</span>'
            f'<h1>{TITLE}</h1><p>{AUTHOR} · {ADVISOR}</p></div></div></div><div id="flow">{flow}</div>')


def result_html(rows) -> str:
    import html

    if not rows:
        return '<p class="muted">Tải video lên rồi bấm “Đoán ý định”.</p>'
    head = "<tr><th>Đã xem</th><th>C · phân loại</th><th>F · ghép câu</th><th>G · tự viết</th></tr>"
    body = ""
    for r in rows:
        c = f'<b>{html.escape(r["C"])}</b>' if r["C"] else '<span class="muted">chưa có bộ phân loại</span>'
        tip = f' title="độ chênh {r["m"]:.4f}"' if r["m"] == r["m"] else ""
        f = f'<b{tip}>{html.escape(r["F"])}</b>'
        if r["ask"]:
            f += '<span class="ask">robot hỏi lại</span>'
        g = f'<b>{html.escape(r["G"])}</b>' if r["G"] else '<span class="muted">không chạy</span>'
        body += (f'<tr><td>{r["p"]}%</td><td><span class="badge bC">C</span>{c}</td>'
                 f'<td><span class="badge bF">F</span>{f}</td><td><span class="badge bG">G</span>{g}</td></tr>')
    return f'<table class="res">{head}{body}</table>'


def _nbox(text: str) -> str:
    return f'<div class="nbox">{text}</div>'


def build_app(args):
    import html

    import gradio as gr

    models = Models(args.classifier, args.context_mean)
    tasks = models.tasks

    def analyse(video, start_s, end_s, use_g):
        if not video:
            return result_html([]), "", gr.update(), []
        frames, times, start, end = read_video(video, start_s, end_s)
        emb = models.image_emb(frames)
        rows, gallery, last = [], [], {}
        for p in PCTS:
            pos = S15.sample8(times, S15.human_end(start, end, p))
            seq, fr8 = emb[pos], frames[pos]
            c = models.predict_c(seq)
            f = models.predict_f(seq)
            g = models.predict_g(fr8) if use_g else ""
            rows.append({"p": p, "C": c[1] if c else None, "F": f[1], "m": f[2], "ask": f[3], "G": g})
            if p == 50:
                gallery = [(fr8[i], f"{times[pos[i]]:.1f}s") for i in range(len(fr8))]
            last = {"C": c[1] if c else None, "F": f[1], "G": g or None}
        choices = [s for s in (last["G"], last["C"], last["F"]) if s]
        note = _nbox(f"🎬 Đoạn hành động <b>{start:.1f}s → {end:.1f}s</b> · 8 khung mỗi mức · F hỏi lại khi không chắc")
        return result_html(rows), note, gr.update(choices=choices, value=choices[0] if choices else None), gallery

    def robot(sentence, scene_task):
        if not sentence:
            return None, _nbox("Chọn hoặc gõ một câu ý định trước.")
        if not args.vlia_checkpoint:
            return None, _nbox("Chưa đặt <b>--vlia-checkpoint</b> khi chạy app.")
        out = Path(tempfile.mkdtemp(prefix="vlia_demo_"))
        cmd = ["bash", str(ROOT / "app/run_rollout.sh"), str(out), str(int(scene_task)), sentence, args.vlia_checkpoint]
        r = subprocess.run(cmd, capture_output=True, text=True)
        vids = sorted(out.glob("videos/*.mp4"))
        recs = sorted(out.glob("episodes/*.json"))
        if r.returncode != 0 or not vids:
            return None, _nbox("Chạy robot lỗi:<pre>" + html.escape((r.stderr or r.stdout)[-2000:]) + "</pre>")
        ok = json.loads(recs[0].read_text())["success"] if recs else None
        return str(vids[0]), _nbox(f"Cảnh LIBERO <b>{int(scene_task)}</b> · kết quả: <b>{'✅ thành công' if ok else '⏳ chưa hoàn thành'}</b>")

    with gr.Blocks(title="VLIA · Demo đồ án") as demo:
        gr.HTML(hero_html())
        with gr.Tab("Bước 1 · Đoán ý định"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_classes="card"):
                    video = gr.Video(label="Video người đang làm", height=440)
                with gr.Column(scale=3, min_width=260, elem_classes="card"):
                    start = gr.Number(value=-1, label="Tay bắt đầu (giây)")
                    end = gr.Number(value=-1, label="Đặt xong (giây)")
                    gr.Markdown("Để -1 nếu video không có đoạn đứng yên.", elem_classes="hint")
                    use_g = gr.Checkbox(value=True, label="Chạy cả G (Qwen tự viết, chậm hơn)")
                    btn = gr.Button("Đoán ý định", elem_classes="primary-btn", size="lg")
            with gr.Column(elem_classes="card"):
                table = gr.HTML(result_html([]))
                note = gr.HTML()
            gallery = gr.Gallery(label="8 khung hình ở mức 50%", columns=8, height=140, elem_classes="card")
        with gr.Tab("Bước 2 · Robot làm theo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, elem_classes="card"):
                    sentence = gr.Dropdown(label="Câu ý định (từ bước 1 hoặc tự gõ)",
                                           allow_custom_value=True)
                    scene = gr.Dropdown(choices=[(f"{t} · {tasks[t]}", t) for t in sorted(tasks)], value=0,
                                        label="Cảnh LIBERO-10")
                    gr.HTML(_nbox("💡 Robot <b>không nhận câu lệnh</b>, chỉ nhận token ý định."))
                    go = gr.Button("Chạy robot", elem_classes="primary-btn", size="lg")
                with gr.Column(scale=3, elem_classes="card"):
                    rvideo = gr.Video(label="Robot trong mô phỏng LIBERO", height=420)
                    rnote = gr.HTML()
        gr.HTML('<div id="foot">Đồ án tốt nghiệp 2026 · Trường Đại học Công nghệ, ĐHQGHN</div>')
        btn.click(analyse, [video, start, end, use_g], [table, note, sentence, gallery])
        go.click(robot, [sentence, scene], [rvideo, rnote])
    return demo


def launch(demo, **kw):
    """Gradio 6 takes theme/css in launch(); older versions in Blocks (set before launch)."""
    import inspect

    import gradio as gr

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", radius_size="lg",
                           font=[gr.themes.GoogleFont("Be Vietnam Pro"), "sans-serif"])
    if "css" in inspect.signature(demo.launch).parameters:
        return demo.launch(theme=theme, css=CSS, **kw)
    demo.theme, demo.css = theme, CSS
    return demo.launch(**kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    d3 = os.environ.get("VLIA_DATA_ROOT", "/media/dhqg/d3")
    ap.add_argument("--classifier", default=str(ROOT / "app/classifier_svm_lin.npz"))
    ap.add_argument("--context-mean", default=f"{d3}/vlia_outputs/stage12/scores.npz",
                    help="npz holding context_mean (saved by run_stage12_e2e.py score)")
    ap.add_argument("--vlia-checkpoint", default=os.environ.get("VLIA_CHECKPOINT", ""))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio link")
    args = ap.parse_args()
    launch(build_app(args), server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
