"""Gradio inference app for the lung disease detector.

Loads a YOLO checkpoint and runs it on an uploaded chest X-ray image.

Weight resolution order:
  1. `MODEL_WEIGHTS` env var, if set and the file exists.
  2. `weights/best.pt` in the repo root, if present (under `runs/detect/train/weights/best.pt` from a
     training run).
  3. Fallback to a stock COCO-pretrained YOLOv8n checkpoint in DEMO MODE,
     so the app is provably working end-to-end even without a trained
     lung-disease model. Demo mode detects everyday objects, not diseases,
     and is labelled as such in the UI.

Run locally:
    pip install -r requirements-app.txt
    python app.py
"""

import os

import gradio as gr
import spaces
from ultralytics import YOLO

LUNG_DISEASE_CLASSES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
    "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "Rib fracture",
]

DEFAULT_WEIGHTS_PATH = "weights/best.pt"

DISCLAIMER = (
    "**Research / educational demo only — not a medical device.** "
    "Do not use these predictions for diagnosis or any clinical decision. "
)


def _resolve_weights() -> tuple[str, bool]:
    """Return (weights_path, is_demo_mode)."""
    env_path = os.environ.get("MODEL_WEIGHTS")
    if env_path and os.path.isfile(env_path):
        return env_path, False
    if os.path.isfile(DEFAULT_WEIGHTS_PATH):
        return DEFAULT_WEIGHTS_PATH, False
    # If no trained lung-disease checkpoint available: fall back to a small
    # general-purpose pretrained model to keep the app functional.
    return "yolov8n.pt", True


WEIGHTS_PATH, DEMO_MODE = _resolve_weights()
model = YOLO(WEIGHTS_PATH)


# Required for Hugging Face Spaces' ZeroGPU hardware
@spaces.GPU
def predict(image):
    if image is None:
        return None, "Upload a chest X-ray image to run detection."

    results = model.predict(source=image, verbose=False)
    result = results[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    if len(result.boxes) == 0:
        summary = "No findings above the confidence threshold."
    else:
        names = result.names
        lines = [
            f"- **{names[int(box.cls)]}** — confidence {float(box.conf):.2f}"
            for box in result.boxes
        ]
        summary = "\n".join(lines)

    return annotated, summary


mode_banner = (
    "⚠️ **Demo mode**: no trained lung-disease checkpoint was found, so this "
    "instance is running a stock COCO-pretrained YOLOv8n model as a pipeline "
    "smoke test. It will detect everyday objects, not lung diseases. Train "
    "a model (see the notebooks in `YOLO/`) and place it at `weights/best.pt` "
    "or set `MODEL_WEIGHTS` to serve real predictions."
    if DEMO_MODE
    else f"Serving trained checkpoint: `{WEIGHTS_PATH}`."
)

with gr.Blocks(title="Lung Disease Detection") as demo:
    gr.Markdown("# Lung Disease Detection (YOLO)")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(mode_banner)

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Chest X-ray")
        image_output = gr.Image(label="Detections")

    findings_output = gr.Markdown(label="Findings")
    run_button = gr.Button("Run detection", variant="primary")

    run_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[image_output, findings_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))