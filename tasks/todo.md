# Implementation Task List: GanodermaScout

## Phase 1: Project Foundation & Storage Infrastructure

### Task 1: Repository structure and core Docker Compose environment
**Description:** Set up the monorepo root structure, root `.env.example`, `.gitignore`, and the foundational Docker Compose stack defining PostgreSQL 16 with pgvector, Redis 7, and MinIO object storage with integrated container healthchecks.

**Acceptance criteria:**
- [ ] Root directory structure created with modular service folders (`services/gateway`, `services/core`, `services/vision`, `services/agents`, `services/worker`, `web`, `data`).
- [ ] `docker-compose.yml` configures PostgreSQL 16 (`pgvector/pgvector:pg16`), Redis 7, and MinIO with persistent volume mappings and healthcheck probes.
- [ ] `.env.example` provides default connection strings, credentials, and service port definitions.

**Verification:**
- [ ] Tests pass: `docker compose config` validates configuration without errors.
- [ ] Build succeeds: `docker compose up -d postgres redis minio` starts all three containers in healthy status.
- [ ] Manual check: Run `docker compose exec postgres psql -U postgres -d ganodermascout -c "SELECT 1;"` and verify response.

**Dependencies:** None

**Files likely touched:**
- `docker-compose.yml`
- `.env.example`
- `.gitignore`
- `Makefile`

**Estimated scope:** Medium (4 files)

---

### Task 2: Relational database schema and migrations
**Description:** Define the SQLAlchemy ORM models and Alembic migration scripts in the Core service for plantation hierarchy (plantations, blocks, palms), inspections, detections, recommendations, document chunks with vector embeddings, and detection corrections.

**Acceptance criteria:**
- [ ] PostgreSQL `vector` extension enabled via initial migration script.
- [ ] Tables created: `plantations`, `blocks`, `palms`, `inspections`, `detections`, `recommendations`, `document_chunks`, and `detection_corrections` with proper foreign key relationships and timestamps.
- [ ] `document_chunks` table includes a 1536-dimensional vector column with an HNSW cosine index.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_models.py` confirms model instantiation and relationship integrity.
- [ ] Build succeeds: `alembic upgrade head` applies cleanly against PostgreSQL.
- [ ] Manual check: Inspect table schemas in PostgreSQL to verify column types, constraints, and HNSW index creation.

**Dependencies:** Task 1

**Files likely touched:**
- `services/core/src/db/base.py`
- `services/core/src/db/models.py`
- `services/core/alembic/env.py`
- `services/core/tests/test_models.py`

**Estimated scope:** Medium (4 files)

---

### Task 3: Seed data and sample datasets
**Description:** Assemble the bundled sample dataset containing annotated *Ganoderma* palm base images (YOLO/COCO format), curated open-access Agronomy text passages on Basal Stem Rot, sample golden evaluation QA pairs, and configure DVC configuration pointing to remote storage for full datasets.

**Acceptance criteria:**
- [ ] `data/sample_dataset/images/` contains sample images representing primordium and mature basidiocarp stages with corresponding label txt/json files.
- [ ] `data/sample_dataset/corpus/` contains open-access agronomy articles and extension bulletins with complete metadata and license records.
- [ ] `data/sample_dataset/eval_golden.json` contains 10 golden evaluation QA pairs with ground-truth citation spans.
- [ ] `.dvc/config` and `.dvcignore` initialized for dataset tracking.

**Verification:**
- [ ] Tests pass: `python scripts/validate_sample_data.py` validates image formats, annotation bounding box coordinates, and corpus files.
- [ ] Build succeeds: `dvc status` executes without errors.
- [ ] Manual check: Confirm sample images can be read by PIL/OpenCV and labels match image dimensions.

**Dependencies:** Task 1

**Files likely touched:**
- `scripts/validate_sample_data.py`
- `.dvc/config`
- `data/sample_dataset/corpus/manifest.json`
- `data/sample_dataset/eval_golden.json`

**Estimated scope:** Medium (4 files)

---

### Checkpoint: Foundation
- [ ] Core storage containers (Postgres, Redis, MinIO) boot healthy.
- [ ] Database schema and pgvector indexes applied cleanly.
- [ ] Sample datasets and verification scripts pass.

---

## Phase 2: Vision Service & Model Serving

### Task 4: Image quality inspection filter
**Description:** Implement an image pre-processing quality gate using OpenCV to evaluate incoming palm photos for blur (Laplacian variance), extreme exposure (underexposed/overexposed histogram clipping), and minimum resolution before invoking heavy neural net detectors.

**Acceptance criteria:**
- [ ] Quality inspector function evaluates image buffer and returns a structured quality report (`is_acceptable`, `blur_score`, `exposure_score`, `rejection_reason`).
- [ ] Rejects images with Laplacian variance $< 100$ with rejection reason `image_blurred`.
- [ ] Rejects images with severe over/under-exposure with descriptive rejection reasons.

**Verification:**
- [ ] Tests pass: `pytest services/vision/tests/test_quality_filter.py` with sharp, blurred, dark, and bright test fixtures.
- [ ] Build succeeds: `ruff check services/vision/` passes with zero lint errors.
- [ ] Manual check: Run filter script against blurry sample image and verify `is_acceptable=False`.

**Dependencies:** Task 1

**Files likely touched:**
- `services/vision/src/quality.py`
- `services/vision/tests/test_quality_filter.py`
- `services/vision/requirements.txt`

**Estimated scope:** Small (3 files)

---

### Task 5: Vision inference service with multi-detector registry
**Description:** Build the FastAPI Vision inference service exposing `/detect` and `/models` endpoints, supporting a model registry capable of loading and switching between YOLOv12, YOLOv13, RT-DETRv3, and RF-DETR models (with mock/fallback weights for lightweight local CPU testing).

**Acceptance criteria:**
- [ ] `/detect` accepts image file and `model_name` query parameter, returning bounding boxes `[x_min, y_min, x_max, y_max]`, class label (`primordium` or `mature_basidiocarp`), confidence score, and inference latency in milliseconds.
- [ ] Image quality check is executed as a prerequisite; failing quality returns HTTP 422 with a structured retake recommendation.
- [ ] `/models` endpoint lists available model architectures and loaded weights status.
- [ ] Dockerfile created for the vision service with CPU/CUDA conditional runtime support.

**Verification:**
- [ ] Tests pass: `pytest services/vision/tests/test_detect_api.py` testing detection outputs and quality rejections.
- [ ] Build succeeds: `docker build -t ganodermascout-vision services/vision` builds cleanly.
- [ ] Manual check: Send a sample image to `POST /detect?model=yolov12` via `curl` and inspect JSON response format.

**Dependencies:** Task 4

**Files likely touched:**
- `services/vision/src/main.py`
- `services/vision/src/detector.py`
- `services/vision/src/schemas.py`
- `services/vision/tests/test_detect_api.py`
- `services/vision/Dockerfile`

**Estimated scope:** Medium (5 files)

---

### Checkpoint: Vision Pipeline
- [ ] Quality inspection filter flags low-quality test images.
- [ ] `/detect` endpoint serves inference and latency metrics across detector models.
- [ ] Bounding boxes, class names, and confidence scores match API schema.

---

## Phase 3: Knowledge Base & Hybrid RAG Engine

### Task 6: Agronomy document ingestion and chunking pipeline
**Description:** Implement the document ingestion module that parses agronomic research documents, reports, and extension guides (Markdown, HTML, PDF), extracts licensing and publication metadata, splits text into semantically coherent chunks, and assigns unique passage identifiers.

**Acceptance criteria:**
- [ ] Ingests documents from `data/sample_dataset/corpus/`, extracting title, author, year, publication source, and license.
- [ ] Splits documents into overlapping text chunks (chunk size: 500 tokens, overlap: 50 tokens) preserving paragraph and section headers.
- [ ] Emits structured chunks with deterministic chunk IDs and source citation anchors.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_ingestion.py` verifying chunking and metadata preservation.
- [ ] Build succeeds: Ingestion runner script parses the sample corpus into JSON chunks without exceptions.
- [ ] Manual check: Verify extracted chunks retain correct publication title, year, and passage offset.

**Dependencies:** Task 2, Task 3

**Files likely touched:**
- `services/core/src/rag/ingestion.py`
- `services/core/src/rag/chunker.py`
- `services/core/tests/test_ingestion.py`

**Estimated scope:** Small (3 files)

---

### Task 7: Dual-mode embedding and pgvector hybrid retrieval with RRF and reranker
**Description:** Build the vector indexer and hybrid retrieval service supporting dual-mode embeddings (OpenAI `text-embedding-3-small` or local HuggingFace `BAAI/bge-small-en-v1.5`), executing hybrid search combining dense pgvector cosine similarity with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF) and an optional cross-encoder reranker.

**Acceptance criteria:**
- [ ] Embedding client generates vectors seamlessly in either cloud API or local mode based on environment variables.
- [ ] Populates `document_chunks` table with embeddings and builds PostgreSQL full-text search vectors.
- [ ] Hybrid retriever implements RRF scoring ($k=60$) over dense and sparse result lists.
- [ ] Optional reranker scores candidate passages and returns top-$k$ ranked passages with source citations.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_retrieval.py` testing dense, sparse, and hybrid search output order.
- [ ] Build succeeds: Database query plan confirms HNSW index usage on dense vector search.
- [ ] Manual check: Run query *"What are the sanitation protocols for mature Ganoderma conks?"* and confirm top retrieved passages contain MPOB sanitation guidelines.

**Dependencies:** Task 6

**Files likely touched:**
- `services/core/src/rag/embedding.py`
- `services/core/src/rag/retriever.py`
- `services/core/src/rag/reranker.py`
- `services/core/tests/test_retrieval.py`

**Estimated scope:** Medium (4 files)

---

### Checkpoint: Knowledge Retrieval
- [ ] Sample corpus parsed, chunked, and embedded into pgvector.
- [ ] Hybrid search with RRF returns relevant agronomic passages.
- [ ] Reranker ranks authoritative management guidelines higher than general mentions.

---

## Phase 4: LangGraph Multi-Agent Workflow

### Task 8: LangGraph stateful agent graph
**Description:** Construct the LangGraph multi-agent decision workflow featuring state management, a Router Agent, Vision Agent coordinator, Memory Step (hydrating palm and block history), Retrieval Agent, and an Answer Synthesis Agent producing structured advice with citations.

**Acceptance criteria:**
- [ ] Defines the shared `AgentState` schema containing image detections, palm/block history, retrieved passages, drafted advisory, and verification status.
- [ ] Router agent classifies requests into `new_inspection`, `history_query`, or `general_advisory`.
- [ ] Retrieval agent queries the hybrid knowledge base using contextual terms from user question and detection classes.
- [ ] Synthesis agent produces recommendations where every factual claim references citation anchors `[Source: ID]`.
- [ ] Supports both cloud LLMs (OpenAI/Anthropic/Gemini) and local LLMs (Ollama/vLLM) via standard LangChain chat model interfaces.

**Verification:**
- [ ] Tests pass: `pytest services/agents/tests/test_graph.py` mocking tool responses and verifying state transitions.
- [ ] Build succeeds: `python -m services.agents.src.graph --dry-run` executes graph topology validation.
- [ ] Manual check: Execute graph with a mock mature basidiocarp detection and verify output suggests sanitation and isolation trenching with citations.

**Dependencies:** Task 5, Task 7

**Files likely touched:**
- `services/agents/src/state.py`
- `services/agents/src/graph.py`
- `services/agents/src/nodes.py`
- `services/agents/src/llm.py`
- `services/agents/tests/test_graph.py`

**Estimated scope:** Medium (5 files)

---

### Task 9: Verification agent guardrails
**Description:** Implement the specialized Verification Agent node in the LangGraph workflow that acts as a strict guardrail: validating citation anchors against retrieved passages, enforcing the complete ban on hallucinated or unsupported chemical dosages, rejecting prompt injection attempts, and rewriting ungrounded outputs.

**Acceptance criteria:**
- [ ] Parses every citation tag `[Source: ID]` in drafted advice and verifies that referenced passages exist in state; flags ungrounded claims.
- [ ] Scans response for chemical dosage figures (e.g., "apply X grams of hexaconazole"); if the exact dosage is not found verbatim in retrieved sources, triggers automated redaction or rewrite.
- [ ] Evaluates input prompts against adversarial patterns (prompt injection / jailbreaks) and halts execution with a safe refusal when detected.
- [ ] Verification state approved, rewritten, or refused is recorded in state metadata.

**Verification:**
- [ ] Tests pass: `pytest services/agents/tests/test_verifier.py` with safety violation test cases, dosage hallucinations, and injection attempts.
- [ ] Build succeeds: `ruff check services/agents/` passes without warnings.
- [ ] Manual check: Feed draft advice with an invented chemical dosage and confirm the verifier removes it or rewrites to require laboratory/agronomist guidance.

**Dependencies:** Task 8

**Files likely touched:**
- `services/agents/src/verifier.py`
- `services/agents/src/safety_rules.py`
- `services/agents/tests/test_verifier.py`

**Estimated scope:** Small (3 files)

---

### Checkpoint: Multi-Agent Synthesis
- [ ] LangGraph graph executes end-to-end through router, retrieval, synthesis, and verification.
- [ ] Verification agent successfully blocks ungrounded chemical dosage claims.
- [ ] Output advice cites valid source documents.

---

## Phase 5: Core API & Asynchronous Job Queue

### Task 10: Asynchronous inspection worker and Redis Streams queue
**Description:** Build the asynchronous background worker service using Redis Streams (`stream:inspection_jobs`) to process inspection tasks, invoke vision inference and the multi-agent graph, update database state, and handle retries with dead-letter queue routing (`stream:inspection_dlq`).

**Acceptance criteria:**
- [ ] Worker joins a Redis consumer group and polls for pending inspection jobs.
- [ ] Executes pipeline: downloads image from MinIO, calls Vision service, invokes LangGraph agent, writes detections and recommendations to PostgreSQL.
- [ ] Implements exponential backoff retry (up to 3 attempts) and routes unrecoverable failures to the dead-letter stream with error diagnostic context.
- [ ] Dockerfile created for the worker service.

**Verification:**
- [ ] Tests pass: `pytest services/worker/tests/test_worker.py` simulating job lifecycle and error retries.
- [ ] Build succeeds: `docker build -t ganodermascout-worker services/worker` builds cleanly.
- [ ] Manual check: Push a test job to Redis stream via CLI and verify worker consumes, processes, and acknowledges the message.

**Dependencies:** Task 2, Task 5, Task 8

**Files likely touched:**
- `services/worker/src/consumer.py`
- `services/worker/src/pipeline.py`
- `services/worker/tests/test_worker.py`
- `services/worker/Dockerfile`

**Estimated scope:** Medium (4 files)

---

### Task 11: Core API endpoints, MinIO image upload, and Server-Sent Events (SSE) stream
**Description:** Implement Core FastAPI endpoints for submitting inspection jobs (`POST /api/v1/inspections`), streaming job progress via Server-Sent Events (`GET /api/v1/inspections/{job_id}/events`), uploading images to MinIO, and fetching completed inspection details.

**Acceptance criteria:**
- [ ] `POST /api/v1/inspections` validates multipart image upload, stores image in MinIO bucket `ganodermascout-images`, inserts database record, pushes job to Redis Streams, and returns HTTP 202 with `job_id`.
- [ ] `GET /api/v1/inspections/{job_id}/events` provides an SSE stream emitting status transitions (`queued` $\to$ `analyzing` $\to$ `retrieving` $\to$ `verifying` $\to$ `completed`).
- [ ] `GET /api/v1/inspections/{job_id}` returns full inspection record including bounding boxes, confidence scores, cited recommendation, and image URL.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_inspections_api.py` validating upload, SSE stream, and result retrieval.
- [ ] Build succeeds: FastAPI OpenAPI docs (`/docs`) generate valid OpenAPI 3.1 specification.
- [ ] Manual check: Upload image via `curl`, connect to SSE endpoint, and watch events transition to completed.

**Dependencies:** Task 2, Task 10

**Files likely touched:**
- `services/core/src/api/v1/inspections.py`
- `services/core/src/services/storage.py`
- `services/core/src/services/events.py`
- `services/core/tests/test_inspections_api.py`

**Estimated scope:** Medium (4 files)

---

### Task 12: Palm & block historical progression timeline API with agronomist correction endpoints
**Description:** Implement Core API endpoints for tracking disease progression over time across individual palms and plantation blocks, along with agronomist feedback endpoints for correcting model detections.

**Acceptance criteria:**
- [ ] `GET /api/v1/palms/{palm_id}/timeline` returns chronological inspections with bounding boxes and stage transitions.
- [ ] `GET /api/v1/blocks/{block_id}/summary` returns aggregated counts of healthy vs. infected palms, stage breakdown, and 30/60/90-day incidence velocity.
- [ ] `POST /api/v1/inspections/{inspection_id}/corrections` records human agronomist bounding box/stage corrections into `detection_corrections` table.
- [ ] `GET /api/v1/corrections/export` exports corrected annotations in COCO/YOLO format for future fine-tuning.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_history_and_corrections.py` validating timeline aggregations and correction exports.
- [ ] Build succeeds: `ruff check services/core/` passes with zero errors.
- [ ] Manual check: Seed 3 inspections for palm P-101 and verify timeline returns all 3 in descending chronological order.

**Dependencies:** Task 2, Task 11

**Files likely touched:**
- `services/core/src/api/v1/history.py`
- `services/core/src/api/v1/corrections.py`
- `services/core/tests/test_history_and_corrections.py`

**Estimated scope:** Small (3 files)

---

### Checkpoint: Core Backend & Jobs
- [ ] Inspection upload successfully places file in MinIO and enqueues job in Redis.
- [ ] Worker processes job and SSE delivers live progress to client.
- [ ] Historical timelines and agronomist corrections endpoints return expected schemas.

---

## Phase 6: Gateway & Automation Integration

### Task 13: Go API Gateway
**Description:** Build the high-performance Go API Gateway service providing OAuth 2.0 PKCE authentication, JWT validation, token-bucket rate limiting per client IP/token, reverse proxy routing to Core API, and signed webhook dispatching.

**Acceptance criteria:**
- [ ] Reverse proxies `/api/*` requests to the Core FastAPI service with minimal latency overhead ($< 10\text{ms}$).
- [ ] Validates JWT Bearer tokens and checks scopes (`scout:write`, `agronomist:review`, `admin:read`).
- [ ] Enforces token-bucket rate limiting (configurable requests per minute, returning HTTP 429 when exhausted).
- [ ] Dockerfile created for the Go Gateway service.

**Verification:**
- [ ] Tests pass: `go test -v ./services/gateway/...` validating auth middleware and rate limiting.
- [ ] Build succeeds: `go build -o bin/gateway ./services/gateway/cmd/server` compiles cleanly.
- [ ] Manual check: Send unauthenticated request to protected route and verify HTTP 401 Unauthorized; send rapid burst to verify HTTP 429.

**Dependencies:** Task 11

**Files likely touched:**
- `services/gateway/cmd/server/main.go`
- `services/gateway/internal/auth/jwt.go`
- `services/gateway/internal/middleware/ratelimit.go`
- `services/gateway/internal/proxy/reverse.go`
- `services/gateway/Dockerfile`

**Estimated scope:** Medium (5 files)

---

### Task 14: n8n event automation workflows for critical infection alerts
**Description:** Configure automated alert workflows in n8n triggered by Core API webhooks when mature basidiocarps are detected or block-level infection density exceeds threshold limits, routing notifications to Slack and Email channels.

**Acceptance criteria:**
- [ ] n8n workflow JSON definition created in `deploy/n8n/workflows/critical_ganoderma_alert.json`.
- [ ] Core API dispatches HMAC-signed webhook to n8n webhook listener on `inspection.completed` when mature basidiocarp is present.
- [ ] Workflow formats message containing block ID, palm ID, confidence score, image link, and recommended emergency sanitation steps.
- [ ] Includes mock webhook receiver test in local development environment.

**Verification:**
- [ ] Tests pass: `pytest services/core/tests/test_webhook_dispatcher.py` verifying HMAC signature and payload.
- [ ] Build succeeds: n8n workflow file is valid JSON conforming to n8n workflow schema.
- [ ] Manual check: Trigger mature detection event and verify n8n webhook node logs receipt and formats message payload.

**Dependencies:** Task 11

**Files likely touched:**
- `deploy/n8n/workflows/critical_ganoderma_alert.json`
- `services/core/src/services/webhooks.py`
- `services/core/tests/test_webhook_dispatcher.py`

**Estimated scope:** Small (3 files)

---

### Checkpoint: Gateway & Alerting
- [ ] Go Gateway successfully proxies authenticated traffic and rate-limits excess requests.
- [ ] Webhook alerts dispatch on mature basidiocarp detection.

---

## Phase 7: Evaluation Harness & Benchmark Suite

### Task 15: Evaluation harness for detectors, RAG modes, and memory toggle
**Description:** Develop the benchmark execution harness capable of running batch evaluations on a site-disjoint test set across the 4 detector models, 3 retrieval configurations, and memory toggle, logging metrics to MLflow and local JSON summaries.

**Acceptance criteria:**
- [ ] Runs detection evaluation computing mAP@50, mAP@50:95, and per-class Recall for YOLOv12, YOLOv13, RT-DETRv3, and RF-DETR.
- [ ] Runs RAG evaluation over golden test set computing Hit@5, Recall@5, and MRR.
- [ ] Runs Faithfulness and Citation Precision evaluation using deterministic LLM rubric.
- [ ] Runs adversarial safety suite computing block rate against prohibited claims and injection attacks.
- [ ] Logs parameters and metric artifacts to MLflow experiment tracking.

**Verification:**
- [ ] Tests pass: `pytest evaluation/tests/test_eval_metrics.py` testing calculation of mAP, MRR, and faithfulness.
- [ ] Build succeeds: `python evaluation/run_benchmark.py --mode smoke` completes without error on sample data.
- [ ] Manual check: Verify MLflow UI or generated summary JSON contains complete metric entries.

**Dependencies:** Task 5, Task 7, Task 8, Task 9

**Files likely touched:**
- `evaluation/run_benchmark.py`
- `evaluation/metrics/detection.py`
- `evaluation/metrics/rag.py`
- `evaluation/metrics/safety.py`
- `evaluation/tests/test_eval_metrics.py`

**Estimated scope:** Medium (5 files)

---

### Task 16: Benchmark metrics reporter and Pareto frontier generator
**Description:** Implement the benchmark reporting service that computes the speed-accuracy Pareto frontier (mAP@50 vs. latency in ms) across detector models and publishes a structured benchmark summary consumed by the frontend benchmark page.

**Acceptance criteria:**
- [ ] Aggregates latest benchmark runs into a standardized `benchmark_summary.json`.
- [ ] Computes Pareto-optimal frontier coordinates comparing detector mAP@50 against inference latency.
- [ ] Core API exposes `GET /api/v1/benchmark/summary` serving this data.

**Verification:**
- [ ] Tests pass: `pytest evaluation/tests/test_pareto.py` verifying Pareto frontier identification logic.
- [ ] Build succeeds: `GET /api/v1/benchmark/summary` returns valid JSON conforming to benchmark schema.
- [ ] Manual check: Inspect Pareto coordinates to verify models on the frontier are correctly flagged.

**Dependencies:** Task 15

**Files likely touched:**
- `evaluation/reporter.py`
- `services/core/src/api/v1/benchmark.py`
- `evaluation/tests/test_pareto.py`

**Estimated scope:** Small (3 files)

---

### Checkpoint: Benchmarking
- [ ] Evaluation harness executes across all models and retrieval strategies.
- [ ] MLflow tracks runs and metrics.
- [ ] Pareto frontier generated and exposed via API.

---

## Phase 8: Frontend Application

### Task 17: Next.js upload interface with real-time SSE progress and image quality preview
**Description:** Build the Next.js 15 capture and inspection submission view featuring image upload with client-side drag-and-drop, block/palm selector, question input, pre-upload blur check indicator, and live SSE progress timeline.

**Acceptance criteria:**
- [ ] Form allows selecting image, block ID, palm ID, question text, detector model, and memory toggle.
- [ ] Shows client-side image preview and fast HTML5 canvas blur warning if image is severely unfocused.
- [ ] Submits to `/api/v1/inspections` and connects to SSE stream displaying live progress step transitions (`queued` $\to$ `analyzing` $\to$ `retrieving` $\to$ `verifying` $\to$ `completed`).

**Verification:**
- [ ] Tests pass: `npm test web/tests/UploadForm.test.tsx` passes with React Testing Library.
- [ ] Build succeeds: `npm run build` inside `web/` completes with 0 TypeScript and lint errors.
- [ ] Manual check: Upload an image in browser, verify SSE spinner updates, and redirects to result page on completion.

**Dependencies:** Task 11, Task 13

**Files likely touched:**
- `web/src/app/upload/page.tsx`
- `web/src/components/UploadForm.tsx`
- `web/src/components/JobProgress.tsx`
- `web/tests/UploadForm.test.tsx`

**Estimated scope:** Medium (4 files)

---

### Task 18: Interactive detection viewer with bounding box canvas, confidence slider, and cited advice
**Description:** Develop the inspection result view in Next.js with an interactive HTML5 canvas overlay rendering color-coded bounding boxes for primordia and mature basidiocarps, a dynamic confidence threshold slider, and a grounded recommendation card with clickable citation anchors.

**Acceptance criteria:**
- [ ] Renders uploaded image with accurately scaled bounding box overlays based on original image dimensions.
- [ ] Interactive slider dynamically filters visible boxes based on confidence threshold without network reload.
- [ ] Displays explicit disclaimer when no detections exceed threshold.
- [ ] Recommendation text displays clickable citation badges that open a popover or drawer with the exact source passage title and excerpt.

**Verification:**
- [ ] Tests pass: `npm test web/tests/DetectionViewer.test.tsx` testing box filtering and citation interactions.
- [ ] Build succeeds: Next.js builds clean without hydration mismatches.
- [ ] Manual check: Move confidence slider from 0.1 to 0.9 and observe boxes dynamically appearing and disappearing.

**Dependencies:** Task 17

**Files likely touched:**
- `web/src/app/inspections/[id]/page.tsx`
- `web/src/components/DetectionCanvas.tsx`
- `web/src/components/RecommendationCard.tsx`
- `web/src/components/CitationDrawer.tsx`
- `web/tests/DetectionViewer.test.tsx`

**Estimated scope:** Medium (5 files)

---

### Task 19: Palm/block timeline history view and benchmark dashboard
**Description:** Build the historical progression timeline view for palms/blocks and the benchmark dashboard visualising detector Pareto frontier charts (Recharts) and comparative RAG metrics tables.

**Acceptance criteria:**
- [ ] `/palms/[id]` renders a chronological timeline card deck showing historical inspection dates, stages, and thumbnails.
- [ ] `/blocks/[id]` renders an estate map/card view with aggregated infected counts and stage ratios.
- [ ] `/benchmark` displays an interactive Pareto frontier scatter plot (mAP@50 vs. Latency) and comparison tables for retrieval modes (Dense vs. Hybrid vs. Rerank).

**Verification:**
- [ ] Tests pass: `npm test web/tests/BenchmarkDashboard.test.tsx` testing chart rendering and data table mapping.
- [ ] Build succeeds: `npm run build` completes cleanly.
- [ ] Manual check: Navigate to `/benchmark` and verify Pareto chart points hover to show model names and latency values.

**Dependencies:** Task 12, Task 16, Task 18

**Files likely touched:**
- `web/src/app/palms/[id]/page.tsx`
- `web/src/app/blocks/[id]/page.tsx`
- `web/src/app/benchmark/page.tsx`
- `web/src/components/ParetoChart.tsx`
- `web/tests/BenchmarkDashboard.test.tsx`

**Estimated scope:** Medium (5 files)

---

### Checkpoint: Frontend UI
- [ ] Upload flow with live SSE transitions completes smoothly.
- [ ] Interactive bounding box canvas and confidence slider update in real time.
- [ ] Historical timelines and benchmark Pareto charts render cleanly.

---

## Phase 9: Observability, Packaging & E2E Verification

### Task 20: Observability stack
**Description:** Integrate Prometheus metric instrumentation across Go Gateway, Core API, and Vision service, and provision Grafana dashboards tracking request throughput, p95 latencies, Redis queue depth, and model confidence distributions.

**Acceptance criteria:**
- [ ] Each microservice exposes `/metrics` in Prometheus format.
- [ ] `deploy/prometheus/prometheus.yml` configures scrape targets for all services.
- [ ] `deploy/grafana/dashboards/` contains pre-provisioned dashboard JSON displaying latency, error rates, queue backlog, and detection counts.

**Verification:**
- [ ] Tests pass: Querying `http://localhost:9090/api/v1/targets` confirms all microservice endpoints are UP.
- [ ] Build succeeds: Grafana dashboard JSON passes syntax validation.
- [ ] Manual check: Run inspection requests and verify Grafana graphs show incoming request spikes and latency metrics.

**Dependencies:** Task 5, Task 11, Task 13

**Files likely touched:**
- `deploy/prometheus/prometheus.yml`
- `deploy/grafana/dashboards/ganodermascout_overview.json`
- `services/core/src/metrics.py`
- `services/vision/src/metrics.py`

**Estimated scope:** Medium (4 files)

---

### Task 21: Single-command `docker compose up` orchestration and end-to-end smoke test suite
**Description:** Orchestrate the complete multi-service stack in `docker-compose.yml` with dependency health ordering, write the automated end-to-end smoke test script, and verify single-command startup from cold clone to completed inspection.

**Acceptance criteria:**
- [ ] Single command `docker compose up -d` boots Gateway, Core API, Vision, Agents, Worker, Postgres, Redis, MinIO, and Web frontend with no manual intervention.
- [ ] `scripts/e2e_smoke_test.sh` executes an end-to-end test: uploads sample image, waits for SSE completion, verifies bounding boxes, and checks cited recommendation.
- [ ] Root `README.md` updated with architecture diagram, prerequisites, and quick-start instructions.

**Verification:**
- [ ] Tests pass: `./scripts/e2e_smoke_test.sh` exits with code 0.
- [ ] Build succeeds: `docker compose build` succeeds with zero errors across all images.
- [ ] Manual check: Run full cold startup, upload photo via browser, and view cited result.

**Dependencies:** Task 1 through Task 20

**Files likely touched:**
- `docker-compose.yml`
- `scripts/e2e_smoke_test.sh`
- `README.md`
- `Makefile`

**Estimated scope:** Medium (4 files)

---

### Checkpoint: Final Release Readiness
- [ ] Single `docker compose up -d` boots entire system into healthy state.
- [ ] Automated end-to-end smoke test passes 100% green.
- [ ] Complete portfolio demo ready for review.
