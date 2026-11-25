import sys
import types

class _NoOpModule(types.ModuleType):
    def __getattr__(self, name):
        return None

sys.modules["executorch.extension.llm.custom_ops"] = _NoOpModule("executorch.extension.llm.custom_ops")
sys.modules["executorch.extension.llm.custom_ops.custom_ops"] = _NoOpModule("executorch.extension.llm.custom_ops.custom_ops")

import argparse
import ast
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# ==========================================
# 1. PCB INSPECTION RULES
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
                "impact": "Detection confidence is low.",
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
# 2. YOLOv8 ONNX VISION DETECTOR (LETTERBOX)
# ==========================================
class PCBDetector:
    def __init__(self, onnx_path="pcb.onnx", conf_thresh=0.25, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
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
        print(f"[Vision] Loaded {onnx_path} ({self.img_w}x{self.img_h}) | Classes: {self.class_names}")

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

    def predict(self, frame):
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
            if conf >= self.conf_thresh:
                cx, cy, bw, bh = row[:4]
                x1 = int(max(0, (cx - bw / 2.0 - pad_x) / r))
                y1 = int(max(0, (cy - bh / 2.0 - pad_y) / r))
                box_w = int(min(w - x1, bw / r))
                box_h = int(min(h - y1, bh / r))
                boxes.append([x1, y1, box_w, box_h])
                scores.append(conf)
                class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)
        detections = []
        annotated = frame.copy()

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                label = self.class_names[class_ids[idx]]
                conf = scores[idx]
                detections.append({"defect": label, "confidence": conf, "box": [x, y, bw, bh]})

                color = (0, 0, 255) if conf >= 0.50 else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(
                    annotated, f"{label} {conf:.2f}", (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

        latency_ms = (time.time() - t0) * 1000.0
        return detections, annotated, latency_ms


# ==========================================
# 3. OFFLINE EXECUTORCH LLAMA ADVISOR
# ==========================================
class EdgeLlamaAdvisor:
    def __init__(self, model_dir="./llama32_1b_pte"):
        t0 = time.time()
        print("[LLM] Loading quantized Llama 3.2 1B via ExecuTorch (offline)...")
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
# 4. CSV AUDIT LOGGER & MAIN PIPELINE
# ==========================================
def log_inspection(csv_path, source, findings, rule_decision, vision_ms, llm_sec, tps, advisory):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
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


def run_single_frame(frame, source_name, detector, advisor, csv_path):
    detections, annotated, vision_ms = detector.predict(frame)
    findings = assess_detections(detections)
    rule_decision = get_rule_decision(findings)

    print(f"\n[Vision] {len(findings)} detection(s) in {vision_ms:.1f} ms | Rule Decision: {rule_decision}")
    for f in findings:
        print(f"  - {f['defect']} (conf={f['confidence']:.2f}, severity={f['severity']})")

    if advisor is None:
        return annotated

    if not findings:
        advisory = (
            "No visual defects detected. Board passes visual screening; "
            "proceed to electrical continuity testing (PASS_TO_NEXT_TEST)."
        )
        llm_sec, tps = 0.0, 0.0
        print("\n--- Fast Rule Decision (0.00s | LLM bypassed for clean board) ---")
        print(advisory)
        print("------------------------------------------------------------------")
    else:
        advisory, llm_sec, tps = advisor.advise(findings, rule_decision)
        print(f"\n--- Offline Llama QC Advisory ({llm_sec:.2f}s | {tps:.2f} tok/s) ---")
        print(advisory)
        print("------------------------------------------------------------------")

    log_inspection(csv_path, source_name, findings, rule_decision, vision_ms, llm_sec, tps, advisory)
    print(f"[Log] Saved record to {csv_path}")
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Edge PCB Defect Inspector + Offline Llama QC")
    parser.add_argument("--image", type=str, help="Path to a PCB image (works over SSH without display)")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g., 0 for USB webcam)")
    parser.add_argument("--onnx", type=str, default="pcb.onnx", help="Path to YOLOv8 ONNX model")
    parser.add_argument("--llm", type=str, default="./llama32_1b_pte", help="Path to ExecuTorch model folder")
    parser.add_argument("--csv", type=str, default="inspection_logs.csv", help="Output CSV log file")
    parser.add_argument("--vision-only", action="store_true", help="Skip LLM for instant vision verification")
    args = parser.parse_args()

    detector = PCBDetector(onnx_path=args.onnx)
    advisor = None if args.vision_only else EdgeLlamaAdvisor(model_dir=args.llm)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        annotated = run_single_frame(frame, args.image, detector, advisor, args.csv)
        cv2.imwrite("annotated_output.jpg", annotated)
        print("[Saved] Annotated image written to annotated_output.jpg")

    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera}")
        print("[Live] Camera active. Press 'a' to run Llama QC analysis on current frame, 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections, annotated, vision_ms = detector.predict(frame)
            cv2.putText(
                annotated, f"FPS: {1000/max(1,vision_ms):.1f} | Press 'a' for AI QC",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Edge PCB Inspector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                run_single_frame(frame, f"camera_{args.camera}", detector, advisor, args.csv)
            elif key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Specify --image <path_to_pcb.jpg> or --camera 0")


if __name__ == "__main__":
    main()
