> Note: **NOT for medical purposes.** This project is for research and educational purposes only. Predictions must never be used for real diagnosis or clinical decisions.

# Lung Disease Detection with YOLO

This notebook is inspired by [VinBigData-CXR-AD YOLOv5 14 Class train](https://www.kaggle.com/code/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-train?kernelSessionId=52422980) and attempts to perform training on the whole dataset, with more disease types than the one in the competition.

The aim was to develop a model that can detect multiple different types of diseases in the lungs. The task is not only classification but also detection using bounding boxes (x and y coordinates). The work serves as a starting point for more powerful and promising future models.

Different strategies were used to deal with the dataset to arrive at a "ground truth" as 3 different radiologists had annotated each image. Only the most successful version is mentioned here in which the annotations from 2 of the 3 radiologists had to agree with averaged out bounding boxes.

---
## References

- **Kaggle Notebook**: [VinBigData-CXR-AD YOLOv5 14 Class train](https://www.kaggle.com/code/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-train?kernelSessionId=52422980)  
- **Original Dataset**: [VinDr-CXR on PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/)  
- **Smaller Test Dataset (resized 256x256 PNG)**: [VinBigData Chest X-ray Resized on Kaggle](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-256x256)  

---

## Repo layout

```
YOLO/                   Training notebooks + Ultralytics dataset configs (yolov8, yolov12)
dataset/                Curated CSV annotations (raw images are gitignored — see below)
rescale_images.py       CLI: rescale bbox annotations to match resized images
visualize_patient_details.py  Quick look at patient metadata (patient_details.npy)
app.py                  Gradio inference app (serves a trained YOLO checkpoint)
requirements.txt        Notebook dependencies
requirements-app.txt    Minimal deps for serving the app (kept separate for fast deploys)
Dockerfile              Docker Container build for the inference app
weights/                Weight checkpoints (not committed)
```

## Running the app locally

```bash
pip install -r requirements-app.txt
python app.py
```

Opens on `http://localhost:7860`. Without a trained checkpoint at `weights/best.pt` (or `MODEL_WEIGHTS` env var pointing elsewhere), it runs in **demo mode** with a stock COCO-pretrained YOLOv8n model, just to prove the upload → inference → annotated-image pipeline works end to end. Train a model from the notebooks in `YOLO/` and drop the resulting `best.pt` into `weights/` to serve real predictions.

After training, copy `runs/detect/train/weights/best.pt` to `weights/best.pt` in this repo (or point `MODEL_WEIGHTS` at it) to serve it through `app.py`.

## Deploying

App is deployed on the **Gradio app on [Hugging Face Spaces](https://huggingface.co/spaces)** (free tier: 2 vCPU / 16GB RAM). To replicate:

1. Create a new Space → SDK: **Gradio**.
2. Push this repo's `app.py`, `requirements-app.txt` (rename to `requirements.txt` inside the Space, or point the Space's build at it) and, if you have one, `weights/best.pt` to the Space's git repo.
3. The Space builds and serves automatically at `https://huggingface.co/spaces/<user>/<space>`.

**Alternative: Docker anywhere.** The included `Dockerfile` builds a self-contained image (`docker build -t lung-detector . && docker run -p 7860:7860 lung-detector`). This same image can be deployed to the Hugging Face Spaces (SDK: Docker), with full control over the environment.

---