# GanodermaScout

GanodermaScout helps plantation staff inspect oil palms for Ganoderma boninense. A user uploads a photo of a palm base and asks a question. The system detects fruiting bodies, labels each as a primordium or a mature basidiocarp, and returns management advice that cites its sources. It stores each inspection by palm and block, so it can show how infection changes over time and alert an agronomist when a block crosses a threshold.

The repo also contains an evaluation harness. It compares four detectors (YOLOv12, YOLOv13, RT-DETRv3, RF-DETR), three retrieval modes (dense, hybrid, hybrid with reranker), and memory on versus off, and publishes the results on a benchmark page.

A detection is decision support and not a diagnosis. Fruiting bodies are generally a late sign, so the system never reports a palm as healthy because nothing was detected.

## Vision service

The FastAPI service exposes `POST /detect` (multipart field `file`, query parameter `model_name`) and `GET /models`. Supported model names: `yolov12`, `yolov13`, `rtdetrv3`, and `rf-detr`. Detection responses contain `bbox` coordinates in `[x_min, y_min, x_max, y_max]` order, `class_label`, `confidence`, and `inference_latency_ms`. Quality rejections return HTTP 422 with a retake recommendation. Configure trained checkpoint paths with `VISION_YOLOV12_WEIGHTS`, `VISION_YOLOV13_WEIGHTS`, `VISION_RTDETRV3_WEIGHTS`, and `VISION_RFDETR_WEIGHTS`.

Without configured weights, `/detect` returns HTTP 503 rather than inventing detections. For CPU/API smoke tests, set `VISION_MOCK_MODELS=true`; mock models return no detections and `/models` marks their status as `mock`. The Docker build defaults to a lightweight CPU image. Set `INSTALL_MODEL_RUNTIMES=true` to install PyTorch, Ultralytics, and RF-DETR; CUDA builds must provide a compatible CUDA-enabled Python/PyTorch base image and matching `PYTORCH_INDEX_URL`.
