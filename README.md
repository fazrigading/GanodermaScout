# GanodermaScout

GanodermaScout helps plantation staff inspect oil palms for Ganoderma boninense. A user uploads a photo of a palm base and asks a question. The system detects fruiting bodies, labels each as a primordium or a mature basidiocarp, and returns management advice that cites its sources. It stores each inspection by palm and block, so it can show how infection changes over time and alert an agronomist when a block crosses a threshold.

The repo also contains an evaluation harness. It compares four detectors (YOLOv12, YOLOv13, RT-DETRv3, RF-DETR), three retrieval modes (dense, hybrid, hybrid with reranker), and memory on versus off, and publishes the results on a benchmark page.

A detection is decision support and not a diagnosis. Fruiting bodies are generally a late sign, so the system never reports a palm as healthy because nothing was detected.

## Vision service

The FastAPI service exposes `POST /detect` (multipart field `file`, query parameter `model_name`) and `GET /models`. Supported model names: `yolov12`, `yolov13`, `rtdetrv3`, and `rf-detr`. Detection responses contain `bbox` coordinates in `[x_min, y_min, x_max, y_max]` order, `class_label`, `confidence`, and `inference_latency_ms`. Quality rejections return HTTP 422 with a retake recommendation. Configure trained checkpoint paths with `VISION_YOLOV12_WEIGHTS`, `VISION_YOLOV13_WEIGHTS`, `VISION_RTDETRV3_WEIGHTS`, and `VISION_RFDETR_WEIGHTS`.

Without configured weights, `/detect` returns HTTP 503 rather than inventing detections. For CPU/API smoke tests, set `VISION_MOCK_MODELS=true`; mock models return no detections and `/models` marks their status as `mock`. The Docker build defaults to a lightweight CPU image. Set `INSTALL_MODEL_RUNTIMES=true` to install PyTorch, Ultralytics, and RF-DETR; CUDA builds must provide a compatible CUDA-enabled Python/PyTorch base image and matching `PYTORCH_INDEX_URL`.

## Document ingestion

The Core RAG ingestion module parses Markdown, HTML, and PDF documents from `data/sample_dataset/corpus/`, using `manifest.json` values when available and embedded document metadata otherwise. It writes JSON passage chunks to stdout by default; run `python -m services.core.src.rag.ingestion --corpus data/sample_dataset/corpus --output chunks.json` to save a file. Chunks use the `cl100k_base` tokenizer, default to 500 tokens with 50-token overlap for long sections, retain document/section headings, and include deterministic citation anchors and extracted-text offsets.

## Hybrid retrieval

The Core RAG indexer stores passage text, metadata, and 1536-dimensional embeddings in PostgreSQL; a generated English `tsvector` and GIN index support sparse search, while the existing HNSW cosine index serves dense search. `EMBEDDING_PROVIDER=huggingface` uses `EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`); `EMBEDDING_PROVIDER=openai` uses `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`) and `OPENAI_API_KEY`. Local 384-dimensional BGE vectors are zero-padded to the database's 1536 dimensions without changing cosine similarity. Retrieval combines dense and sparse rankings with RRF ($k=60$); optionally pass `CrossEncoderReranker` to rerank candidates.

Apply the Core Alembic migrations before indexing. Run `python -m services.core.src.rag.indexer --corpus data/sample_dataset/corpus` with `DATABASE_URL` and the embedding configuration set to populate the index. `hybrid_retrieve` returns ranked passages with stable citation anchors.

## Stateful agent graph

The LangGraph workflow routes inspections, history queries, and general advice; new-image requests call the Vision service, memory nodes load palm/block inspection history from PostgreSQL, and retrieval adds detection classes to the user query before dense/sparse search. Structured synthesis attaches only citation anchors present in retrieved passages.

An input guard refuses known prompt-injection requests before vision, memory, or retrieval calls. A final verifier approves only citations found in retrieved passages and rewrites ungrounded claims or chemical dosage phrases not found verbatim in those passages. Verification status (`approved`, `rewritten`, or `refused`) is recorded in graph state.

Configure `LLM_PROVIDER` as `openai`, `anthropic`, `gemini`, `ollama`, or `vllm`; provider SDKs implement LangChain chat-model interfaces. Run `python -m services.agents.src.graph --dry-run` to validate graph topology without contacting providers or databases.
