import sys
import types

# Prevent loading ARMv8.2 AOT export library on Raspberry Pi 4 Cortex-A72
class _NoOpModule(types.ModuleType):
    def __getattr__(self, name):
        return None

sys.modules["executorch.extension.llm.custom_ops"] = _NoOpModule("executorch.extension.llm.custom_ops")
sys.modules["executorch.extension.llm.custom_ops.custom_ops"] = _NoOpModule("executorch.extension.llm.custom_ops.custom_ops")

import ast
import base64
import csv
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import onnxruntime as ort
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# ==========================================
# 1. PCB INSPECTION RULES & GUARDRAILS
# ==========================================
DEFECT_RULES = {
    "open_circuit": {
        "severity": "HIGH",
        "impact": "May interrupt the electrical path or signal.",
        "action": "Perform electrical continuity testing."
    },
    "short": {
        "severity": "HIGH",
        "impact": "May connect separate electrical paths unintentionally.",
        "action": "Perform short-circuit verification."
    },
    "spur": {
        "severity": "MEDIUM",
        "impact": "May reduce electrical clearance.",
        "action": "Inspect conductor spacing and PCB design."
    },
    "spurious_copper": {
        "severity": "MEDIUM",
        "impact": "Unwanted copper may cause clearance or connection issues.",
        "action": "Inspect the affected copper area."
    },
    "missing_hole": {
        "severity": "HIGH",
        "impact": "May affect component mounting or electrical connection.",
        "action": "Compare with the PCB design."
    },
    "mouse_bite": {
        "severity": "MEDIUM",
        "impact": "May affect the board edge or nearby copper.",
        "action": "Inspect the affected edge and conductor clearance."
    }
}

FALLBACK_CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper"
]

CSV_PATH = os.path.expanduser("~/inspection_logs.csv")
SAMPLE_IMAGE_PATH = os.path.expanduser("~/missinghole.jpeg")


def assess_detections(detections):
    findings = []
    for item in detections:
        defect = item["defect"].lower().replace(" ", "_")
        confidence = round(float(item["confidence"]), 3)
        rule = DEFECT_RULES.get(defect)

        if confidence < 0.50:
            findings.append({
                "defect": defect,
                "confidence": confidence,
                "severity": "UNCERTAIN",
                "impact": "Detection confidence is below 0.50 threshold.",
                "action": "Capture another image and verify manually."
            })
        elif rule:
            findings.append({
                "defect": defect,
                "confidence": confidence,
                "severity": rule["severity"],
                "impact": rule["impact"],
                "action": rule["action"]
            })
        else:
            findings.append({
                "defect": defect,
                "confidence": confidence,
                "severity": "UNKNOWN",
                "impact": "No verified inspection rule is available.",
                "action": "Manual engineering inspection required."
            })
    return findings


def get_rule_decision(findings):
    if not findings:
        return "PASS_TO_NEXT_TEST"
    severities = {f["severity"] for f in findings}
    if "UNCERTAIN" in severities or "UNKNOWN" in severities:
        return "HOLD"
    if "HIGH" in severities:
        return "REJECT"
    return "HOLD"


def create_ai_prompt(findings, rule_decision):
    summary_lines = "\n".join(
        f"- {f['defect']} (conf={f['confidence']:.2f}, severity={f['severity']}): "
        f"{f['impact']} Next: {f['action']}"
        for f in findings
    )
    user_msg = f"""You are a PCB QC assistant. Findings:
{summary_lines}
Rule Decision: {rule_decision}

Reply in 4 short bullet points (under 50 words total):
- Problem:
- PCB Effect:
- Next Check:
- Decision ({rule_decision}): simple reason.
Do not invent faults or assume visual detection proves electrical failure."""

    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ==========================================
# 2. USB WEBCAM CAPTURE THREAD (V4L2)
# ==========================================
class USBCameraManager:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
        self.device_index = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self._connect()

    def _connect(self):
        for idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                self.cap = cap
                self.device_index = idx
                with self.lock:
                    self.latest_frame = frame
                self.running = True
                t = threading.Thread(target=self._reader_loop, daemon=True)
                t.start()
                print(f"[Camera] Connected USB webcam at /dev/video{idx} ({frame.shape[1]}x{frame.shape[0]})")
                return
            cap.release()
        print("[Camera] No USB webcam detected (upload/sample modes still active).")

    def _reader_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.05)

    def is_available(self):
        return self.running and self.cap is not None

    def get_frame(self):
        if not self.is_available():
            self._connect()
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


# ==========================================
# 3. YOLOv8 ONNX DETECTOR & EXECUTORCH LLM
# ==========================================
class PCBDetector:
    def __init__(self, onnx_path=os.path.expanduser("~/pcb.onnx"), iou_thresh=0.45):
        self.iou_thresh = iou_thresh
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.img_h = shape[2] if isinstance(shape[2], int) else 640
        self.img_w = shape[3] if isinstance(shape[3], int) else 640

        meta = self.sess.get_modelmeta().custom_metadata_map
        if "names" in meta:
            names_dict = ast.literal_eval(meta["names"])
            self.class_names = [names_dict[i] for i in sorted(names_dict.keys())]
        else:
            self.class_names = FALLBACK_CLASSES
        print(f"[Vision] Ready: {onnx_path} ({self.img_w}x{self.img_h})")

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.img_w / w, self.img_h / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        dw = (self.img_w - new_w) / 2.0
        dh = (self.img_h - new_h) / 2.0
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, left, top

    def predict(self, frame, conf_thresh=0.50):
        t0 = time.time()
        h, w = frame.shape[:2]
        padded, r, pad_x, pad_y = self._letterbox(frame)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        preds = self.sess.run(None, {self.input_name: blob})[0][0].T
        boxes, scores, class_ids = [], [], []

        for row in preds:
            cls_scores = row[4:]
            cid = int(np.argmax(cls_scores))
            conf = float(cls_scores[cid])
            if conf >= conf_thresh:
                cx, cy, bw, bh = row[:4]
                x1 = int(max(0, (cx - bw / 2.0 - pad_x) / r))
                y1 = int(max(0, (cy - bh / 2.0 - pad_y) / r))
                box_w = int(min(w - x1, bw / r))
                box_h = int(min(h - y1, bh / r))
                boxes.append([x1, y1, box_w, box_h])
                scores.append(conf)
                class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, self.iou_thresh)
        detections = []
        annotated = frame.copy()

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                label = self.class_names[class_ids[idx]]
                conf = scores[idx]
                detections.append({"defect": label, "confidence": conf, "box": [x, y, bw, bh]})

                color = (40, 40, 240) if conf >= 0.50 else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)
                tag = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated, (x, max(0, y - th - 10)), (x + tw + 8, y), color, -1)
                cv2.putText(
                    annotated, tag, (x + 4, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )

        latency_ms = (time.time() - t0) * 1000.0
        return detections, annotated, latency_ms


class EdgeLlamaAdvisor:
    def __init__(self, model_dir=os.path.expanduser("~/llama32_1b_pte")):
        t0 = time.time()
        print("[LLM] Loading quantized Llama 3.2 1B via ExecuTorch...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = ExecuTorchModelForCausalLM.from_pretrained(model_dir)
        print(f"[LLM] Ready in {time.time() - t0:.1f}s")

    def advise(self, findings, rule_decision, max_new_tokens=65):
        prompt = create_ai_prompt(findings, rule_decision)
        in_ids = self.tokenizer.encode(prompt)
        max_seq_len = min(512, len(in_ids) + max_new_tokens)

        t0 = time.time()
        full_text = self.model.text_generation(
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_seq_len=max_seq_len
        )
        elapsed = time.time() - t0

        if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
            reply = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            reply = full_text[len(prompt):]
        reply = reply.replace("<|eot_id|>", "").strip()

        out_tokens = max(1, len(self.tokenizer.encode(reply)))
        tps = out_tokens / max(0.01, elapsed)
        return reply, elapsed, tps


# ==========================================
# 4. LOGGING & EMBEDDED WEB HMI
# ==========================================
def log_inspection(source, findings, rule_decision, vision_ms, llm_sec, tps, advisory):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "source", "defect_count", "findings_summary",
                "rule_decision", "vision_ms", "llm_sec", "tokens_per_sec", "ai_advisory"
            ])
        summary = "; ".join(f"{x['defect']}({x['confidence']:.2f},{x['severity']})" for x in findings) or "NONE"
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            source,
            len(findings),
            summary,
            rule_decision,
            f"{vision_ms:.1f}",
            f"{llm_sec:.2f}",
            f"{tps:.2f}",
            advisory.replace("\n", " | ")
        ])


def read_recent_logs(limit=8):
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return list(reversed(rows[-limit:]))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdgePCB-AI | Offline Edge QC Console</title>
<style>
  :root {
    --bg: #0b1120; --card: #111827; --border: #1f2937; --text: #f3f4f6;
    --muted: #9ca3af; --accent: #38bdf8; --pass: #10b981; --hold: #f59e0b; --reject: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 20px; max-width: 1320px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
           background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 22px; margin-bottom: 20px; }
  .title h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: 0.3px; }
  .title p { font-size: 0.82rem; color: var(--muted); margin-top: 3px; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge { font-size: 0.75rem; font-weight: 600; padding: 5px 10px; border-radius: 999px; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; }
  .badge.offline { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.4); color: #34d399; }
  .grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 20px; margin-bottom: 20px; }
  @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .card h2 { font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--accent); margin-bottom: 14px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  button, .file-btn {
    background: #1e293b; color: var(--text); border: 1px solid #334155; padding: 8px 13px;
    border-radius: 8px; font-size: 0.83rem; font-weight: 600; cursor: pointer; transition: 0.15s;
  }
  button:hover, .file-btn:hover { border-color: var(--accent); }
  button.cam-btn { background: rgba(16,185,129,0.16); border-color: #10b981; color: #34d399; }
  button.primary { background: #0284c7; border-color: #0ea5e9; color: #fff; }
  button.primary:hover { background: #0369a1; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  input[type="file"] { display: none; }
  .slider-row { display: flex; align-items: center; gap: 10px; font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }
  .viewport {
    background: #060913; border: 1px dashed #334155; border-radius: 10px; min-height: 330px;
    display: flex; align-items: center; justify-content: center; overflow: hidden; position: relative;
  }
  .viewport img { max-width: 100%; max-height: 440px; display: block; }
  .live-pill {
    position: absolute; top: 10px; left: 10px; background: rgba(239,68,68,0.9); color: #fff;
    font-size: 0.72rem; font-weight: 700; padding: 4px 9px; border-radius: 6px; letter-spacing: 0.5px;
  }
  .placeholder { color: var(--muted); font-size: 0.9rem; text-align: center; padding: 20px; }
  .decision-banner {
    padding: 14px 16px; border-radius: 10px; font-weight: 700; font-size: 1.05rem;
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
    background: #1e293b; border: 1px solid #334155;
  }
  .decision-PASS_TO_NEXT_TEST { background: rgba(16,185,129,0.15); border-color: var(--pass); color: #34d399; }
  .decision-HOLD { background: rgba(245,158,11,0.15); border-color: var(--hold); color: #fbbf24; }
  .decision-REJECT { background: rgba(239,68,68,0.15); border-color: var(--reject); color: #f87171; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 14px; }
  .metric { background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
  .metric .lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }
  .metric .val { font-size: 1rem; font-weight: 700; margin-top: 3px; color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 0.74rem; text-transform: uppercase; }
  .sev-HIGH { color: #f87171; font-weight: 700; }
  .sev-MEDIUM { color: #fbbf24; font-weight: 700; }
  .sev-UNCERTAIN { color: #94a3b8; font-weight: 700; }
  .llm-box {
    background: #060913; border: 1px solid #1e293b; border-radius: 10px; padding: 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.83rem;
    line-height: 1.55; white-space: pre-wrap; color: #e2e8f0; min-height: 135px; margin-top: 12px;
  }
  .status-bar { font-size: 0.8rem; color: var(--accent); margin-bottom: 10px; min-height: 18px; }
</style>
</head>
<body>

<header>
  <div class="title">
    <h1>EdgePCB-AI • Autonomous Edge Quality Control</h1>
    <p>Bharat AI-SoC 2026–27 (PS5) • Live USB Camera + YOLOv8 ONNX + Guardrails + ExecuTorch Llama 3.2 1B</p>
  </div>
  <div class="badges">
    <span class="badge offline">● 100% OFFLINE MODE</span>
    <span class="badge" id="camBadge">USB Camera: Checking...</span>
    <span class="badge">Raspberry Pi 4 • Arm Cortex-A72</span>
  </div>
</header>

<div class="grid">
  <div class="card">
    <h2>1. Live USB Camera & PCB Inspection Viewport</h2>
    <div class="controls">
      <button class="cam-btn" onclick="startLiveCam()">● Live USB Webcam</button>
      <label class="file-btn">
        Upload Image
        <input type="file" id="fileInput" accept="image/*">
      </label>
      <button onclick="loadSample()">Sample PCB</button>
      <button onclick="runInspect(false)" id="btnVision">Fast Vision Check</button>
      <button class="primary" onclick="runInspect(true)" id="btnFull">Run Full AI QC (YOLO + Llama)</button>
    </div>
    <div class="slider-row">
      <span>Confidence Gate: <strong id="confVal">0.50</strong></span>
      <input type="range" id="confSlider" min="0.15" max="0.85" step="0.05" value="0.50"
             oninput="document.getElementById('confVal').innerText=parseFloat(this.value).toFixed(2)">
      <span>(< 0.50 triggers UNCERTAIN → HOLD)</span>
    </div>
    <div class="status-bar" id="statusText">Ready. Click "Live USB Webcam" to preview camera or select an image.</div>
    <div class="viewport" id="viewport">
      <div class="placeholder">Click "Live USB Webcam" to start camera feed.</div>
    </div>
  </div>

  <div class="card">
    <h2>2. Neuro-Symbolic Decision & Offline Llama Advisory</h2>
    <div class="decision-banner" id="decisionBanner">
      <span>BOARD STATUS</span>
      <span id="decisionText">AWAITING SCAN</span>
    </div>

    <div class="metrics">
      <div class="metric"><div class="lbl">Defects</div><div class="val" id="mDefects">-</div></div>
      <div class="metric"><div class="lbl">Vision Time</div><div class="val" id="mVision">-</div></div>
      <div class="metric"><div class="lbl">Llama Time</div><div class="val" id="mLlm">-</div></div>
      <div class="metric"><div class="lbl">Token Rate</div><div class="val" id="mTps">-</div></div>
    </div>

    <table>
      <thead><tr><th>Defect Class</th><th>Conf</th><th>Severity</th><th>Deterministic Rule Action</th></tr></thead>
      <tbody id="findingsBody">
        <tr><td colspan="4" style="color:var(--muted)">No detections yet.</td></tr>
      </tbody>
    </table>

    <h2 style="margin-top:16px;">3. Offline Llama 3.2 1B Operator Advisory</h2>
    <div class="llm-box" id="advisoryBox">Llama 3.2 1B (ExecuTorch) standing by locally on Arm CPU...</div>
  </div>
</div>

<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
    <h2 style="margin:0;">4. Persistent Inspection Audit Log (inspection_logs.csv)</h2>
    <a href="/download/csv"><button>Download CSV Audit Log</button></a>
  </div>
  <table>
    <thead>
      <tr><th>Timestamp</th><th>Source</th><th>Count</th><th>Findings</th><th>Decision</th><th>Vision (ms)</th><th>LLM (s)</th><th>Tok/s</th></tr>
    </thead>
    <tbody id="logsBody"></tbody>
  </table>
</div>

<script>
let mode = "none"; // "webcam" | "upload" | "sample"
let currentImageB64 = null;
let currentFilename = "uploaded_pcb.jpg";
let timerHandle = null;

function startLiveCam() {
  mode = "webcam";
  currentImageB64 = null;
  currentFilename = "usb_webcam_live";
  document.getElementById('viewport').innerHTML = `
    <span class="live-pill">● LIVE USB FEED</span>
    <img src="/video_feed?t=${Date.now()}" alt="Live USB Webcam">
  `;
  document.getElementById('statusText').innerText = 'Live USB webcam active. Position PCB and click "Fast Vision Check" or "Run Full AI QC".';
}

document.getElementById('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  mode = "upload";
  currentFilename = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    currentImageB64 = reader.result;
    document.getElementById('viewport').innerHTML = `<img src="${currentImageB64}">`;
    document.getElementById('statusText').innerText = `Loaded ${currentFilename}. Ready for inspection.`;
  };
  reader.readAsDataURL(file);
});

function loadSample() {
  mode = "sample";
  currentImageB64 = null;
  currentFilename = "missinghole.jpeg";
  document.getElementById('viewport').innerHTML = `<img src="/sample/image?t=${Date.now()}">`;
  document.getElementById('statusText').innerText = 'Loaded sample board (missinghole.jpeg). Ready for inspection.';
}

async function runInspect(useLlm) {
  if (mode === "none") {
    startLiveCam();
  }
  const btnFull = document.getElementById('btnFull');
  const btnVis = document.getElementById('btnVision');
  btnFull.disabled = true;
  btnVis.disabled = true;

  const t0 = Date.now();
  const statusEl = document.getElementById('statusText');
  timerHandle = setInterval(() => {
    const sec = ((Date.now() - t0) / 1000).toFixed(1);
    statusEl.innerText = useLlm
      ? `Capturing frame & running YOLOv8 + Offline Llama 3.2 1B on Pi 4 CPU... (${sec}s)`
      : `Capturing frame & running YOLOv8 ONNX inference... (${sec}s)`;
  }, 200);

  try {
    const res = await fetch('/api/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: mode,
        image_b64: currentImageB64,
        filename: currentFilename,
        conf: parseFloat(document.getElementById('confSlider').value),
        use_llm: useLlm
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    document.getElementById('viewport').innerHTML = `<img src="${data.annotated_b64}">`;
    const banner = document.getElementById('decisionBanner');
    banner.className = `decision-banner decision-${data.rule_decision}`;
    document.getElementById('decisionText').innerText = data.rule_decision;

    document.getElementById('mDefects').innerText = data.findings.length;
    document.getElementById('mVision').innerText = `${data.vision_ms.toFixed(1)} ms`;
    document.getElementById('mLlm').innerText = data.llm_sec > 0 ? `${data.llm_sec.toFixed(1)} s` : 'Bypassed';
    document.getElementById('mTps').innerText = data.tps > 0 ? `${data.tps.toFixed(2)}/s` : '-';

    const tbody = document.getElementById('findingsBody');
    if (data.findings.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#34d399">No visual defects above threshold.</td></tr>';
    } else {
      tbody.innerHTML = data.findings.map(f => `
        <tr>
          <td><strong>${f.defect}</strong></td>
          <td>${f.confidence.toFixed(2)}</td>
          <td class="sev-${f.severity}">${f.severity}</td>
          <td>${f.action}</td>
        </tr>
      `).join('');
    }

    document.getElementById('advisoryBox').innerText = data.advisory;
    renderLogs(data.logs);
    statusEl.innerText = `Inspection complete in ${((Date.now() - t0) / 1000).toFixed(1)}s. Click "Live USB Webcam" to resume feed.`;
  } catch (err) {
    statusEl.innerText = `Error: ${err.message}`;
  } finally {
    clearInterval(timerHandle);
    btnFull.disabled = false;
    btnVis.disabled = false;
  }
}

function renderLogs(logs) {
  const tbody = document.getElementById('logsBody');
  if (!logs || logs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted)">No records logged yet.</td></tr>';
    return;
  }
  tbody.innerHTML = logs.map(r => `
    <tr>
      <td>${r.timestamp || ''}</td>
      <td>${r.source || ''}</td>
      <td>${r.defect_count || '0'}</td>
      <td>${r.findings_summary || 'NONE'}</td>
      <td><strong>${r.rule_decision || ''}</strong></td>
      <td>${r.vision_ms || ''}</td>
      <td>${r.llm_sec || ''}</td>
      <td>${r.tokens_per_sec || ''}</td>
    </tr>
  `).join('');
}

fetch('/api/status').then(r => r.json()).then(d => {
  const badge = document.getElementById('camBadge');
  if (d.camera_available) {
    badge.innerText = `USB Camera: /dev/video${d.camera_index} Ready`;
    startLiveCam();
  } else {
    badge.innerText = 'USB Camera: Not Detected';
  }
  renderLogs(d.logs);
});
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    detector = None
    advisor = None
    camera = None
    lock = threading.Lock()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/video_feed"):
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = self.camera.get_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    jpg = buf.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.08)  # ~12 FPS smooth preview with low CPU load
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/api/status":
            self._send_json({
                "camera_available": self.camera.is_available(),
                "camera_index": self.camera.device_index,
                "logs": read_recent_logs()
            })

        elif self.path.startswith("/sample/image"):
            if not os.path.isfile(SAMPLE_IMAGE_PATH):
                self.send_error(404, "missinghole.jpeg not found on Pi")
                return
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path == "/download/csv":
            if not os.path.isfile(CSV_PATH):
                self.send_error(404, "inspection_logs.csv not found")
                return
            with open(CSV_PATH, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=inspection_logs.csv")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/inspect":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        mode = payload.get("mode", "webcam")
        conf = float(payload.get("conf", 0.50))
        use_llm = bool(payload.get("use_llm", True))
        filename = payload.get("filename", "usb_webcam_live")

        if mode == "webcam":
            frame = self.camera.get_frame()
            if frame is None:
                self._send_json({"error": "USB webcam not available. Check USB connection."}, status=400)
                return
            filename = f"usb_webcam_video{self.camera.device_index}"
        elif mode == "sample":
            frame = cv2.imread(SAMPLE_IMAGE_PATH)
        else:
            b64_data = payload.get("image_b64", "").split(",")[-1]
            img_bytes = base64.b64decode(b64_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            self._send_json({"error": "Could not capture or decode PCB image."}, status=400)
            return

        with self.lock:
            detections, annotated, vision_ms = self.detector.predict(frame, conf_thresh=conf)
            findings = assess_detections(detections)
            rule_decision = get_rule_decision(findings)

            if not findings:
                advisory = (
                    "No visual defects detected above confidence gate. "
                    "Board passes visual screening; proceed to electrical continuity testing (PASS_TO_NEXT_TEST)."
                )
                llm_sec, tps = 0.0, 0.0
            elif not use_llm:
                advisory = (
                    f"Vision-only scan complete ({rule_decision}). "
                    "Click 'Run Full AI QC (YOLO + Llama)' to generate offline root-cause reasoning."
                )
                llm_sec, tps = 0.0, 0.0
            else:
                advisory, llm_sec, tps = self.advisor.advise(findings, rule_decision)

            cv2.imwrite(os.path.expanduser("~/annotated_output.jpg"), annotated)
            log_inspection(filename, findings, rule_decision, vision_ms, llm_sec, tps, advisory)

        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

        self._send_json({
            "annotated_b64": annotated_b64,
            "findings": findings,
            "rule_decision": rule_decision,
            "vision_ms": vision_ms,
            "llm_sec": llm_sec,
            "tps": tps,
            "advisory": advisory,
            "logs": read_recent_logs()
        })

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    RequestHandler.camera = USBCameraManager()
    RequestHandler.detector = PCBDetector()
    RequestHandler.advisor = EdgeLlamaAdvisor()
    server = ThreadingHTTPServer(("0.0.0.0", 8000), RequestHandler)
    print("\n[UI] EdgePCB-AI Live Dashboard running at: http://192.168.137.198:8000")
    print("[UI] Press Ctrl+C to stop.\n")
    server.serve_forever()
