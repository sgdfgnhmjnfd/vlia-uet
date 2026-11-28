#!/usr/bin/env python3
"""Stage 22 (HABIT-Supervisor, S7414): blind labelling tool for two annotators.

Runs on the lab machine, Python standard library only (no pip install).
  python3 habit_label_tool.py --dir <folder with label_S7414_*.csv> [--host 127.0.0.1] [--port 8765] [--who trang|hung]

- Loads ONLY the chosen annotator's CSV; never reads, shows or serves the other one.
- Never shows anchors / robot-reach estimates / any automatic label.
- Writes the CSV after every change (atomic replace), keeps the same columns and order.
- Serves only the clip/still paths listed in that annotator's CSV (with HTTP Range, so seeking works).
- One timestamped backup of the CSV per annotator per start, in --backup-dir (outside the repo).
"""
import argparse
import csv
import datetime
import io
import json
import os
import re
import shutil
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ANNOTATORS = {"trang": "label_S7414_trang.csv", "hung": "label_S7414_hung.csv"}
NAMES = {"trang": "Trang", "hung": "Hưng"}
COLS = ["order", "episode_index", "clip", "still", "k_from_robot_left", "unclear", "note"]

ARGS = None
LOCK = threading.Lock()
CACHE = {}  # who -> list of row dicts


def csv_path(who):
    return os.path.join(ARGS.dir, ANNOTATORS[who])


def load(who):
    """Load (once) the annotator's own CSV. Caller holds LOCK."""
    if who in CACHE:
        return CACHE[who]
    p = csv_path(who)
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != COLS:
            raise SystemExit("Unexpected columns in %s: %s" % (p, rd.fieldnames))
        rows = list(rd)
    os.makedirs(ARGS.backup_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(p, os.path.join(ARGS.backup_dir, "%s.%s.bak" % (os.path.basename(p), stamp)))
    CACHE[who] = rows
    return rows


def save(who):
    """Atomic write of the annotator's CSV. Caller holds LOCK."""
    p = csv_path(who)
    tmp = p + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)  # default line terminator \r\n, same as the pack script
        w.writeheader()
        w.writerows(CACHE[who])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def is_done(r):
    return r["k_from_robot_left"] in ("1", "2", "3", "4") or r["unclear"] == "1"


def public(r):
    # Only the annotator's own fields + ids. No anchors, no other annotator, no auto labels.
    return {"order": r["order"], "episode": r["episode_index"], "k": r["k_from_robot_left"],
            "unclear": r["unclear"], "note": r["note"]}


class H(BaseHTTPRequestHandler):
    server_version = "habit-label/1"

    def log_message(self, fmt, *a):  # quiet: only errors
        if ARGS.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _who(self, q):
        who = (q.get("who") or [""])[0]
        if who not in ANNOTATORS or (ARGS.who and who != ARGS.who):
            return None
        return who

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            b = PAGE.replace("__LOCKED__", json.dumps(ARGS.who or "")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return
        if u.path == "/api/state":
            who = self._who(q)
            if not who:
                return self._json({"error": "bad annotator"}, 400)
            with LOCK:
                rows = load(who)
                return self._json({"who": who, "name": NAMES[who], "rows": [public(r) for r in rows]})
        m = re.fullmatch(r"/media/(clip|still)/(\d+)", u.path)
        if m:
            who = self._who(q)
            if not who:
                return self._json({"error": "bad annotator"}, 400)
            with LOCK:
                rows = load(who)
                idx = int(m.group(2))
                if not (0 <= idx < len(rows)):
                    return self._json({"error": "no such row"}, 404)
                path = rows[idx][m.group(1)]
            return self._file(path, "video/mp4" if m.group(1) == "clip" else "image/jpeg")
        self._json({"error": "not found"}, 404)

    def _file(self, path, ctype):
        try:
            size = os.path.getsize(path)
            fh = open(path, "rb")
        except OSError:
            return self._json({"error": "missing file: " + os.path.basename(path)}, 404)
        with fh:
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            else:
                self.send_response(200)
            n = end - start + 1
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            fh.seek(start)
            try:
                while n > 0:
                    chunk = fh.read(min(n, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    n -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled a range request while seeking

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/label":
            return self._json({"error": "not found"}, 404)
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            d = json.loads(self.rfile.read(min(ln, 10000)).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._json({"error": "bad json"}, 400)
        who = d.get("who")
        if who not in ANNOTATORS or (ARGS.who and who != ARGS.who):
            return self._json({"error": "bad annotator"}, 400)
        k = str(d.get("k", "")).strip()
        unclear = "1" if str(d.get("unclear", "")).strip() == "1" else ""
        note = re.sub(r"[\r\n,]+", " ", str(d.get("note", ""))).strip()[:200]
        if k not in ("", "1", "2", "3", "4"):
            return self._json({"error": "k must be 1-4 or empty"}, 400)
        if unclear == "1":
            k = ""  # rule in HUONG_DAN_GAN_NHAN.md: unclear -> leave k empty
        with LOCK:
            rows = load(who)
            try:
                idx = int(d.get("idx"))
                r = rows[idx]
            except (TypeError, ValueError, IndexError):
                return self._json({"error": "bad row"}, 400)
            if str(d.get("order")) != r["order"]:
                return self._json({"error": "row mismatch, reload the page"}, 409)
            r["k_from_robot_left"], r["unclear"], r["note"] = k, unclear, note
            try:
                save(who)
            except OSError as e:
                return self._json({"error": "save failed: %s" % e}, 500)
            done = sum(is_done(x) for x in rows)
            return self._json({"ok": True, "row": public(r), "done": done, "total": len(rows)})


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gán nhãn S7414</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#d9d9d4;--acc:#1f6feb;--ok:#1a7f37;--warn:#b35900;--bad:#c62828}
@media (prefers-color-scheme: dark){:root{--bg:#161616;--fg:#ececea;--mut:#a0a09a;--card:#222;--line:#3a3a38;--acc:#4c8dff;--ok:#3fb950;--warn:#e3a14a;--bad:#ff6b6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:12px 16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;flex:1;min-width:160px}.bar>i{display:block;height:100%;background:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:12px}
.vbox{position:relative;width:100%;margin:0 auto;overflow:hidden;background:#000;border-radius:6px}
.vbox video{display:block;width:100%}
.vbox.zoom video{width:200%;max-width:none}
.tag{position:absolute;top:6px;padding:2px 8px;border-radius:4px;background:rgba(0,0,0,.6);color:#fff;font-size:13px;pointer-events:none}
.mid{position:absolute;top:0;bottom:0;left:50%;border-left:2px dashed rgba(255,255,0,.7);pointer-events:none}
.vbox.zoom .mid,.vbox.zoom .tagR{display:none}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.k{font-size:30px;font-weight:700;width:76px;height:68px}
.k.on{background:var(--ok);border-color:var(--ok)}
#unclear.on{background:var(--warn);border-color:var(--warn)}
#robot.on{background:var(--warn);border-color:var(--warn)}
.mut{color:var(--mut)}.small{font-size:13px}
#status{min-width:120px}#status.err{color:var(--bad);font-weight:600}#status.ok{color:var(--ok)}
input[type=text]{font:inherit;padding:7px 9px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);flex:1;min-width:160px}
#grid{display:flex;flex-wrap:wrap;gap:3px}#grid b{width:22px;height:18px;font-size:10px;font-weight:400;display:flex;align-items:center;justify-content:center;border-radius:3px;background:var(--line);cursor:pointer}
#grid b.d{background:var(--ok);color:#fff}#grid b.u{background:var(--warn);color:#fff}#grid b.c{outline:2px solid var(--acc)}
img.still{max-width:100%;border-radius:6px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
#pick{text-align:center;padding:60px 16px}#pick button{font-size:22px;padding:16px 36px;margin:10px}
.hidden{display:none}
</style></head><body>
<div id="pick" class="hidden"><h2>Em là ai?</h2><p class="mut">Công cụ chỉ mở file nhãn của người được chọn. Không mở file của người kia.</p>
<button onclick="go('trang')">Trang</button><button onclick="go('hung')">Hưng</button></div>
<div id="app" class="wrap hidden">
<header><div><b id="who"></b> · clip <b id="pos"></b> · episode <span id="ep" class="mut"></span></div>
<div class="bar"><i id="barfill"></i></div><div><b id="done"></b> đã gán</div><div id="status" class="mut"></div></header>
<div class="card">
 <div id="vbox" class="vbox"><video id="vid" muted playsinline loop preload="auto"></video>
  <div class="mid"></div><div class="tag" style="left:6px">CAMERA ROBOT: đếm donut 1→4 từ trái sang phải</div>
  <div class="tag tagR" style="left:calc(50% + 6px)">camera trên đầu người</div></div>
 <div class="row small">
  <button id="play">⏯ Chạy/dừng <kbd>Space</kbd></button><button id="replay">⟲ Xem lại <kbd>Z</kbd></button>
  <button id="back">◀ 1 khung <kbd>A</kbd></button><button id="fwd">1 khung ▶ <kbd>D</kbd></button>
  <span class="mut">Tốc độ <kbd>S</kbd>:</span><button class="sp" data-r="0.25">0.25×</button><button class="sp" data-r="0.5">0.5×</button><button class="sp" data-r="1">1×</button>
  <button id="zoom">🔍 Phóng nửa trái <kbd>H</kbd></button><button id="stillbtn">🖼 Khung cuối <kbd>F</kbd></button>
  <span class="mut" id="time"></span>
 </div>
 <div id="stillbox" class="hidden" style="margin-top:8px"><img id="still" class="still" alt="khung cuối"></div>
</div>
<div class="card">
 <div class="mut small">Người chỉ vào donut số mấy? (đếm trong <b>nửa trái</b> = camera robot, từ trái sang phải). Nhìn tay người, không nhìn robot.</div>
 <div class="row">
  <button class="k" data-k="1">1</button><button class="k" data-k="2">2</button><button class="k" data-k="3">3</button><button class="k" data-k="4">4</button>
  <button id="unclear">❓ Không chắc <kbd>U</kbd></button>
  <button id="robot">🤖 robot_moving <kbd>R</kbd></button>
 </div>
 <div class="row"><input id="note" type="text" placeholder="Ghi chú thêm (không bắt buộc), Enter để lưu">
  <label class="small"><input type="checkbox" id="auto" checked> Tự sang clip sau khi chọn</label></div>
 <div class="row">
  <button id="prev">← Clip trước</button><button id="next">Clip sau →</button><button id="todo">Tới clip chưa gán <kbd>N</kbd></button>
 </div>
 <div class="mut small" style="margin-top:8px">Phím tắt: <kbd>1</kbd>–<kbd>4</kbd> chọn donut · <kbd>U</kbd> không chắc · <kbd>R</kbd> robot đã với tới (bật trước khi chọn số) · <kbd>←</kbd>/<kbd>→</kbd> clip trước/sau · <kbd>Space</kbd> chạy/dừng · <kbd>Z</kbd> xem lại · <kbd>A</kbd>/<kbd>D</kbd> lùi/tiến 1 khung · <kbd>S</kbd> đổi tốc độ · <kbd>H</kbd> phóng nửa trái · <kbd>F</kbd> khung cuối · <kbd>N</kbd> tới clip chưa gán. Mỗi lần chọn là lưu ngay vào CSV.</div>
</div>
<div class="card"><div class="small mut" style="margin-bottom:6px">Bảng tiến độ (xanh = đã gán, cam = không chắc). Bấm để nhảy tới clip.</div><div id="grid"></div></div>
</div>
<script>
const LOCKED=__LOCKED__;let WHO=null,rows=[],cur=0,speeds=[0.25,0.5,1],rate=0.5,saving=0;
const $=id=>document.getElementById(id),vid=$("vid");
function done(r){return ["1","2","3","4"].includes(r.k)||r.unclear==="1"}
function hasRobot(r){return (r.note||"").split(";").map(s=>s.trim()).includes("robot_moving")}
function extraNote(r){return (r.note||"").split(";").map(s=>s.trim()).filter(s=>s&&s!=="robot_moving").join("; ")}
function go(w){location.hash=w;start(w)}
async function start(w){
  const res=await fetch("/api/state?who="+w);const j=await res.json();
  if(j.error){alert(j.error);return}
  WHO=w;rows=j.rows;$("pick").classList.add("hidden");$("app").classList.remove("hidden");$("who").textContent=j.name;
  document.title="Gán nhãn S7414 · "+j.name;
  const first=rows.findIndex(r=>!done(r));buildGrid();show(first<0?0:first);
}
function buildGrid(){const g=$("grid");g.innerHTML="";rows.forEach((r,i)=>{const b=document.createElement("b");b.textContent=r.order;b.onclick=()=>show(i);g.appendChild(b)});paintGrid()}
function paintGrid(){const bs=$("grid").children;rows.forEach((r,i)=>{bs[i].className=(r.unclear==="1"?"u":done(r)?"d":"")+(i===cur?" c":"")});
  const n=rows.filter(done).length;$("done").textContent=n+"/"+rows.length;$("barfill").style.width=(100*n/rows.length)+"%";
  if(n===rows.length)setStatus("Xong cả "+rows.length+" clip! Báo chị để tính kappa.","ok")}
function show(i){
  if(i<0||i>=rows.length)return;cur=i;const r=rows[i];
  vid.src="/media/clip/"+i+"?who="+WHO;vid.playbackRate=rate;vid.play().catch(()=>{});
  $("still").src="/media/still/"+i+"?who="+WHO;
  $("pos").textContent=r.order+"/"+rows.length;$("ep").textContent=r.episode;
  $("note").value=extraNote(r);paintLabel();paintGrid();
}
function paintLabel(){const r=rows[cur];
  document.querySelectorAll(".k").forEach(b=>b.classList.toggle("on",b.dataset.k===r.k));
  $("unclear").classList.toggle("on",r.unclear==="1");$("robot").classList.toggle("on",hasRobot(r));
  document.querySelectorAll(".sp").forEach(b=>b.classList.toggle("on",+b.dataset.r===rate));}
function setStatus(t,c){const s=$("status");s.textContent=t;s.className=c||"mut"}
function buildNote(robot,extra){return [robot?"robot_moving":"",extra.trim()].filter(Boolean).join("; ")}
async function put(i,k,unclear,note,advance){
  const r=rows[i];saving++;setStatus("Đang lưu…");
  try{const res=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({who:WHO,idx:i,order:r.order,k:k,unclear:unclear,note:note})});
    const j=await res.json();if(!j.ok)throw new Error(j.error||res.status);
    rows[i]=j.row;setStatus("Đã lưu ✓ clip "+r.order,"ok");
  }catch(e){setStatus("LỖI LƯU: "+e.message+" (chưa lưu, thử lại)","err");saving--;return}
  saving--;if(i===cur){paintLabel()}paintGrid();
  if(advance&&$("auto").checked&&i===cur&&cur<rows.length-1)setTimeout(()=>show(cur+1),250);
}
function setK(k){const r=rows[cur];put(cur,String(k),"",buildNote(hasRobot(r),$("note").value),true)}
function toggleUnclear(){const r=rows[cur];const u=r.unclear==="1"?"":"1";put(cur,"",u,buildNote(hasRobot(r),$("note").value),u==="1")}
function toggleRobot(){const r=rows[cur];put(cur,r.k,r.unclear,buildNote(!hasRobot(r),$("note").value),false)}
function saveNote(){const r=rows[cur];put(cur,r.k,r.unclear,buildNote(hasRobot(r),$("note").value),false)}
function step(d){vid.pause();vid.currentTime=Math.max(0,Math.min(vid.duration||0,vid.currentTime+d))}
function setRate(x){rate=x;vid.playbackRate=x;paintLabel()}
function fit(){const vb=$("vbox"),w=vid.videoWidth||16,h=vid.videoHeight||9,a=(vb.classList.contains("zoom")?w/2:w)/h;
  vb.style.width="min(100%, calc(62vh * "+a.toFixed(4)+"))"}
function toggleZoom(){$("vbox").classList.toggle("zoom");fit()}
vid.addEventListener("loadedmetadata",fit);window.addEventListener("resize",fit);
function nextTodo(){for(let s=1;s<=rows.length;s++){const j=(cur+s)%rows.length;if(!done(rows[j])){show(j);return}}setStatus("Không còn clip chưa gán.","ok")}
document.querySelectorAll(".k").forEach(b=>b.onclick=()=>setK(b.dataset.k));
document.querySelectorAll(".sp").forEach(b=>b.onclick=()=>setRate(+b.dataset.r));
$("unclear").onclick=toggleUnclear;$("robot").onclick=toggleRobot;$("prev").onclick=()=>show(cur-1);$("next").onclick=()=>show(cur+1);
$("todo").onclick=nextTodo;$("play").onclick=()=>vid.paused?vid.play():vid.pause();$("replay").onclick=()=>{vid.currentTime=0;vid.play()};
$("back").onclick=()=>step(-0.1);$("fwd").onclick=()=>step(0.1);$("zoom").onclick=toggleZoom;
$("stillbtn").onclick=()=>$("stillbox").classList.toggle("hidden");
$("note").addEventListener("keydown",e=>{if(e.key==="Enter"){saveNote();e.target.blur()}e.stopPropagation()});
vid.addEventListener("timeupdate",()=>{$("time").textContent=(vid.currentTime||0).toFixed(1)+" / "+(vid.duration||0).toFixed(1)+" s"});
vid.addEventListener("error",()=>setStatus("Không mở được video clip "+rows[cur].order,"err"));
document.addEventListener("keydown",e=>{if(!WHO||e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key;
  if(["1","2","3","4"].includes(k))setK(k);else if(k==="u"||k==="U")toggleUnclear();else if(k==="r"||k==="R")toggleRobot();
  else if(k==="ArrowLeft")show(cur-1);else if(k==="ArrowRight")show(cur+1);else if(k===" "){vid.paused?vid.play():vid.pause()}
  else if(k==="z"||k==="Z"){vid.currentTime=0;vid.play()}else if(k==="a"||k==="A")step(-0.1);else if(k==="d"||k==="D")step(0.1);
  else if(k==="s"||k==="S")setRate(speeds[(speeds.indexOf(rate)+1)%speeds.length]);else if(k==="h"||k==="H")toggleZoom();
  else if(k==="f"||k==="F")$("stillbox").classList.toggle("hidden");else if(k==="n"||k==="N")nextTodo();else return;e.preventDefault()});
window.addEventListener("beforeunload",e=>{if(saving>0){e.preventDefault();e.returnValue=""}});
const h=location.hash.slice(1);
if(LOCKED)start(LOCKED);else if(h==="trang"||h==="hung")start(h);else $("pick").classList.remove("hidden");
</script></body></html>
"""


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/media/dhqg/d3/vlia-habit/results/stage22_habit_supervisor",
                    help="folder with label_S7414_trang.csv / label_S7414_hung.csv")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this machine; Tailscale IP = also your laptop")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--who", choices=sorted(ANNOTATORS), help="lock the tool to one annotator")
    ap.add_argument("--backup-dir", default="/media/dhqg/d3/vlia_outputs/stage22_label_backup")
    ap.add_argument("--verbose", action="store_true")
    ARGS = ap.parse_args()
    for w in ([ARGS.who] if ARGS.who else ANNOTATORS):
        if not os.path.isfile(csv_path(w)):
            sys.exit("Không thấy file %s" % csv_path(w))
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), H)
    print("Công cụ gán nhãn đang chạy. Mở trình duyệt: http://%s:%d/%s" % (
        ARGS.host, ARGS.port, ("#" + ARGS.who) if ARGS.who else ""))
    print("CSV: %s | backup: %s | dừng: Ctrl+C" % (ARGS.dir, ARGS.backup_dir))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng. Nhãn đã được lưu sau mỗi lần chọn.")


if __name__ == "__main__":
    main()
