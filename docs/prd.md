# Product Requirements Document: GanodermaScout

- **Version**: 1.0 (Ready for Build)
- **Owner**: Fazri Gading
- **Status**: Approved for Implementation
- **Repository**: [fazrigading/GanodermaScout](https://github.com/fazrigading/GanodermaScout)

---

## 1. Executive Summary

### Problem Statement
Basal Stem Rot (BSR) caused by *Ganoderma boninense* destroys oil palm productivity and tree longevity, yet manual field scouting remains slow, inconsistent, and disconnected from verified agronomic intervention protocols. Field teams lack automated visual staging and reliable, cited management recommendations at the point of inspection.

### Proposed Solution
GanodermaScout is an end-to-end multi-service AI decision-support system that detects and stages *Ganoderma* fruiting bodies (primordium vs. mature basidiocarp) from palm base photographs, tracks historical progression per palm and plantation block, and generates verified agronomic management recommendations grounded strictly in scientific literature via a multi-agent LangGraph workflow.

### Success Criteria
1. **Detection Quality**: Mature basidiocarp Recall@IoU0.5 $\ge 0.88$ and overall mAP@50 $\ge 0.82$ on a site-disjoint holdout test set across benchmarked detector models.
2. **Retrieval & Citation Precision**: Citation precision $\ge 0.90$ with 0% ungrounded chemical dosage hallucinations in recommendation outputs.
3. **Safety Verification**: Safety filter block rate $\ge 0.95$ on adversarial prompts (unsupported chemical dosages, off-label treatments, prompt injections).
4. **Latency & Reliability**: End-to-end p95 processing latency $\le 4.5\text{s}$ (GPU) / $\le 12.0\text{s}$ (8-core CPU); $\ge 99.0\%$ job completion rate across a 500-request load test.
5. **Local Reproducibility**: 100% automated pass rate for smoke tests on local launch via a single `docker compose up` command using bundled sample data.

---

## 2. User Experience & Functionality

### User Personas

| Persona | Role & Context | Primary Need |
| --- | --- | --- |
| **Field Scout** | Plantation worker walking assigned blocks with camera/phone | Rapid capture validation, immediate detection bounding boxes, and field action instructions |
| **Plantation Agronomist** | Estate specialist overseeing disease containment across blocks | Historical infection spread tracking, cited sanitary protocols, and model correction workflows |
| **ML/Platform Reviewer** | Hiring manager or technical evaluator reviewing codebase | Clean architecture verification, benchmark reproducibility, and observable service metrics |

### User Stories & Acceptance Criteria

#### US-1: Palm Photo Submission & Ingestion
*As a Field Scout, I want to submit a palm base photograph with block and palm identifiers so that the tree's health state is inspected.*

- **AC 1.1**: The upload endpoint accepts JPEG and PNG formats up to 15 MB.
- **AC 1.2**: Core API returns an HTTP 202 Accepted with a unique `job_id` within $\le 500\text{ms}$.
- **AC 1.3**: Frontend uploads stream progress and receive status transitions (`queued`, `processing`, `completed`, `failed`) via Server-Sent Events (SSE) without page refresh.
- **AC 1.4**: Client UI validates image capture quality against baseline bounds (blur score via Laplacian variance $> 100$, exposure within dynamic range thresholds) and requests a retake if bounds fail before job queuing.

#### US-2: Visual Detection & Staging
*As a Field Scout or Agronomist, I want to view localized bounding boxes and developmental stages so that I know exactly where and how developed the infection is.*

- **AC 2.1**: The response renders bounding boxes color-coded by class: `primordium` (early stage) and `mature_basidiocarp` (developed conk).
- **AC 2.2**: Each detection displays an explicit confidence score in $[0.00, 1.00]$.
- **AC 2.3**: The UI provides a dynamic confidence threshold slider ($0.10$ to $0.90$ with $0.05$ increments) that updates visible boxes client-side in real time.
- **AC 2.4**: If no fruiting bodies are detected with confidence above the threshold, the system displays an explicit disclaimer: *"No visible fruiting bodies detected. Note: Fruiting bodies are a late-stage symptom of BSR; internal trunk decay may still be active."*

#### US-3: Grounded Agronomic Recommendations
*As a Plantation Agronomist, I want actionable disease management advice citing verified agronomic publications so that I can justify field sanitation decisions.*

- **AC 3.1**: Every factual agronomic claim links to an explicit source citation anchor referencing author, publication title, year, and passage ID.
- **AC 3.2**: Recommendations distinguish actions based on stage: sanitation/mounding/monitoring for primordia versus immediate sanitation, isolation trenching, or controlled de-boling for mature basidiocarps.
- **AC 3.3**: If confidence on detection is $< 0.60$, the recommendation flags the finding for secondary agronomist confirmation before applying physical treatments.

#### US-4: Longitudinal Spread & Palm History
*As a Plantation Agronomist, I want to review previous inspection records for a specific palm and block so that I can evaluate disease progression rate.*

- **AC 4.1**: Palm history view shows an ordered visual timeline of all prior inspections, including dates, detection bounding boxes, and stage transitions.
- **AC 4.2**: Block map/view aggregates total infected palms, distribution by stage, and new incidence count within the last 30/60/90 days.
- **AC 4.3**: When memory toggle is active (`memory_enabled=true`), the recommendation text explicitly references prior inspection states (e.g., *"Palm B-12 transition from primordium observed 42 days ago to mature basidiocarp indicates active fungal expansion"*).

#### US-5: Critical Incidence Alerting
*As a Plantation Agronomist, I want instant alerts when mature fruiting bodies or cluster thresholds are breached so that field containment can begin immediately.*

- **AC 5.1**: Detection of a `mature_basidiocarp` triggers an asynchronous webhook dispatch to n8n within $\le 30\text{s}$ of job completion.
- **AC 5.2**: When block incidence exceeds a configurable limit (default: 3 infected palms within a 5-palm radius), an alert is pushed to configured Slack and email channels via n8n.

#### US-6: Agronomist Feedback & Model Corrections
*As an Agronomist, I want to correct false positives or adjust missed bounding boxes so that the ground-truth benchmark dataset can improve.*

- **AC 6.1**: The UI allows authenticated agronomists to edit, add, or delete bounding boxes and stage tags on any completed inspection.
- **AC 6.2**: Edits are stored in an audit table (`detection_corrections`) with user ID, original detections, corrected annotations, and timestamp.
- **AC 6.3**: An export API (`GET /api/v1/corrections/export`) emits formatted COCO/YOLO annotations ready for DVC versioning and model retraining.

#### US-7: Evaluation Harness & Benchmark Dashboard
*As an ML/Platform Reviewer, I want to compare model families, retrieval pipelines, and memory effects so that I can evaluate system trade-offs.*

- **AC 7.1**: The harness runs automated test suites against a fixed, site-disjoint holdout set for:
  - 4 object detectors: YOLOv12, YOLOv13, RT-DETRv3, RF-DETR.
  - 3 retrieval pipelines: Dense only, Hybrid (BM25 + Dense RRF), Hybrid + Reranker (bge-reranker).
  - Memory toggle: ON vs. OFF.
- **AC 7.2**: The benchmark UI displays a Pareto frontier plot (mAP@50 vs. Inference Latency in ms) comparing all 4 detector models.
- **AC 7.3**: Benchmark page renders comparative metric cards (Recall@50, mAP@50:95, MRR@5, Faithfulness, Citation Precision) with diff toggles against baseline.

#### US-8: Safety & Refusal Enforcement
*As a Platform Reviewer, I want the system to reject harmful or unverified inputs so that no illegal or damaging agricultural practices are recommended.*

- **AC 8.1**: Verification agent checks every drafted recommendation against an explicit safety catalog (banned chemicals, unverified dosage numbers, off-label fungicides).
- **AC 8.2**: If the drafting LLM generates dosage metrics not present in the retrieved passages, the verification agent triggers a strict rewrite removing all numerical dosage references.
- **AC 8.3**: Adversarial prompt injections inside image metadata or user questions are blocked by guardrail filters without altering agent system prompts.

### Non-Goals
- **Non-Goal 1**: Replacing professional laboratory pathogen isolation or certified agronomist diagnosis. GanodermaScout is strictly a decision-support and scouting triage tool.
- **Non-Goal 2**: Asymptomatic or pre-visual early detection (trunk drilling, internal acoustic tomography, hyperspectral satellite/drone sensing).
- **Non-Goal 3**: Native offline mobile applications (iOS/Android) or multi-tenant SaaS billing/subscription management.
- **Non-Goal 4**: Automated chemical ordering, inventory procurement, or autonomous spray machinery dispatch.
- **Non-Goal 5**: Detection of non-*Ganoderma* diseases (e.g., Upper Stem Rot, Curvularia leaf spot) outside of basic out-of-distribution visual rejection.

---

## 3. AI System Requirements

### Model & Tool Requirements

#### 1. Computer Vision Detection Suite
- **Model Registry & Frameworks**: PyTorch, Ultralytics, Hugging Face Transformers, MLflow Model Registry.
- **Detector Architectures**:
  - `YOLOv12` (ultralytics / edge real-time baseline)
  - `YOLOv13` (cutting-edge CNN-attention hybrid)
  - `RT-DETRv3` (real-time vision transformer)
  - `RF-DETR` (receptive-field enhanced detection transformer)
- **Input Resolution**: $640 \times 640$ px RGB normalized.
- **Classes**: `primordium` (class 0), `mature_basidiocarp` (class 1).
- **Pre-filtering**: OpenCV Laplacian variance blur filter (threshold $\sigma^2 < 100$) and luminance histogram clipping check.

#### 2. Agent & LLM Orchestration
- **Agent Framework**: LangGraph (stateful graph-based multi-agent execution).
- **LLM Engine Compatibility (Dual-Mode)**:
  - *Cloud Mode*: OpenAI API (`gpt-4o` / `gpt-4o-mini`) or Anthropic Claude API (`claude-3-5-sonnet`) / Google Gemini API.
  - *Local Mode*: Ollama or vLLM running open-weight models (`llama-3.3-70b`, `qwen2.5-72b` or `qwen2.5-7b-instruct`).
  - Provider selectable via standard environment variables (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`).
- **Embedding Pipeline**:
  - *Cloud Mode*: `text-embedding-3-small` (1536 dims).
  - *Local Mode*: `BAAI/bge-small-en-v1.5` or `nomic-embed-text` (via Hugging Face / Ollama).
  - Stored in `pgvector` with HNSW cosine distance indexing (`m=16`, `ef_construction=64`).
- **Retrieval Pipeline**:
  - Dense vector similarity search via `pgvector`.
  - Sparse lexical search via PostgreSQL Full-Text Search (`tsvector` + `tsquery` using English/agronomy dictionary) or BM25.
  - Reciprocal Rank Fusion (RRF) with constant $k=60$.
  - Optional Cross-Encoder reranker: `BAAI/bge-reranker-base`.

#### 3. Agent Graph Topology

```
                  ┌──────────────┐
                  │ User Request │
                  └──────┬───────┘
                         │
                         ▼
                 [ 1. Router Agent ]
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  [New Inspection] [History Query] [General Inquiry]
         │               │               │
         ▼               │               │
  [2. Vision Agent]      │               │
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
               [ 3. Memory Step ]
            (Loads palm/block history)
                         │
                         ▼
             [ 4. Retrieval Agent ]
         (Dense / Hybrid / Rerank RAG)
                         │
                         ▼
              [ 5. Synthesis Agent ]
            (Drafts cited advisory)
                         │
                         ▼
            [ 6. Verification Agent ]
         (Checks citations, bans hallucinated
          dosages, rejects injection)
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
      [Approved]             [Violations Found]
           │                           │
           │                           ▼
           │                   [Rewrite / Refuse]
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
                  [ Final Output ]
```

### Evaluation Strategy & Benchmarking

#### 1. Datasets & Splits
- **Image Dataset**: Custom annotated dataset of *Ganoderma* basidiocarps and primordia on oil palm bases.
  - *Distribution Policy*: High-quality synthetic/curated sample dataset bundled in repo (`data/sample_dataset/`) for instant smoke testing.
  - *Full Research Dataset*: Tracked via DVC with remote storage pointers to protect unpublished research findings.
  - *Splits*: Site-disjoint train/val/test splits (plantations in the test set never appear in train or validation sets to ensure real-world generalization).
- **Text Corpus**: 150+ open-access Agronomy publications, MPOB (Malaysian Palm Oil Board) guidance, and BSR management extension bulletins with complete CC-BY / open-access licensing metadata.
- **Golden Evaluation Set**: 100 hand-verified image + question pairs with expert reference answers and designated ground-truth citation spans.
- **Adversarial Safety Set**: 30 injection and safety test prompts (off-label chemical inquiries, lethal dosage calculations, prompt override injections).

#### 2. Evaluation Metrics & Standards

| Target Area | Evaluation Metric | Baseline / Target Threshold | Evaluation Method |
| --- | --- | --- | --- |
| **Object Detection** | mAP@50 | $\ge 0.82$ across all classes | COCO eval script on site-disjoint test set |
| **Object Detection** | Recall@50 (Mature) | $\ge 0.88$ | COCO eval script on site-disjoint test set |
| **Detection Speed** | Latency per frame | $\le 45\text{ms}$ (GPU) / $\le 300\text{ms}$ (CPU) | Averaged over 100 consecutive warm runs |
| **Retrieval** | Hit@5, Recall@5, MRR | MRR $\ge 0.75$, Recall@5 $\ge 0.85$ | Evaluated against golden reference passages |
| **Groundedness** | Faithfulness | Score $\ge 0.90$ | LLM-as-a-judge with deterministic rubric |
| **Citation Precision**| Citation Accuracy | $\ge 0.92$ of citations link to valid passages | Programmatic parser matching citations to retrieved chunks |
| **Safety Compliance** | Adversarial Block Rate | $\ge 0.95$ (29/30 minimum) | Automated execution over adversarial test suite |
| **Memory Efficacy** | Delta Answer Quality | $+15\%$ context relevance on follow-ups | Pairwise blind LLM judge comparison (Memory ON vs OFF) |

---

## 4. Technical Specifications

### Architecture Overview

```
                                [ Next.js Frontend ]
                                          │
                               (REST / SSE via OAuth)
                                          ▼
                             [ Go API Gateway (Port 8000) ]
                               (Auth, Rate Limiting, Route)
                                          │
                                          ▼
                       [ Core API Service (Python FastAPI) ]
                                   │              │
                    (Dispatches Job)              (Reads/Writes Data)
                                   ▼                      ▼
                         [ Redis Stream Queue ]  [ PostgreSQL + pgvector ]
                                   │
                                   ▼
                      [ Asynchronous Worker Pool ]
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
      [ Vision Service (FastAPI) ]       [ Agent Service (LangGraph) ]
      (YOLO / RT-DETR / OpenCV)          (Router, RAG, Verifier)
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                           (Job Done Event)
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
     [ Core API DB Callback ]             [ n8n Automation Engine ]
               │                                       │
     (Updates status & SSE)                 (Sends Slack / Email Alerts)
```

### Component Services Catalog

| Component | Stack | Responsibilities |
| --- | --- | --- |
| **API Gateway** | Go (Gin / Chi) | OAuth 2.0 PKCE termination, IP/Token rate limiting (token bucket), reverse proxying, webhook signing. |
| **Core API** | Python (FastAPI, SQLAlchemy) | CRUD for palms/blocks, inspection job creation, database orchestration, SSE streaming. |
| **Vision Inference** | Python (FastAPI, PyTorch) | Model serving for YOLOv12/v13, RT-DETRv3, RF-DETR; image quality validation filter. |
| **Agent Service** | Python (LangGraph) | Multi-agent state graph, memory hydration, hybrid retrieval, synthesis, verification. |
| **Async Worker** | Python (Celery or Redis Streams Consumer) | Pulls inspection jobs from Redis, coordinates vision + agent execution, writes completion state. |
| **Ingestion Pipeline** | Apache Airflow | Document crawler, chunking, embedding generator, vector sync to `pgvector`. |
| **Automation Engine** | n8n | Alert routing workflows (mature basidiocarp alert $\to$ Slack webhook / SMTP notification). |
| **Frontend UI** | Next.js 15 (React 19, TypeScript, Tailwind) | Field capture interface, interactive canvas bounding boxes, timeline charts, benchmark dashboard. |
| **Observability** | Prometheus & Grafana | Scraping metrics from all services (latencies, queue lag, error rates, model confidence). |

### Integration Points & Data Contracts

#### 1. REST API Contract
- `POST /api/v1/inspections`: Accepts `multipart/form-data` with `image` (binary), `block_id` (string), `palm_id` (string), `question` (optional text), `detector_model` (optional enum), `memory_enabled` (boolean). Returns `202 Accepted` with `{"job_id": "uuid", "status": "queued"}`.
- `GET /api/v1/inspections/{job_id}`: Returns current status, bounding boxes, stage determinations, cited recommendations, and execution metadata.
- `GET /api/v1/inspections/{job_id}/events`: Server-Sent Events stream for real-time state updates (`queued` $\to$ `analyzing_image` $\to$ `retrieving_docs` $\to$ `verifying` $\to$ `completed`).
- `GET /api/v1/blocks/{block_id}/timeline`: Returns historical aggregated detection counts by date and stage.
- `GET /api/v1/benchmark/summary`: Returns latest test set evaluation metrics across detector and RAG configurations.

#### 2. Data Stores Schema
- **PostgreSQL 16 with pgvector extension**:
  - `plantations`, `blocks`, `palms`: Physical hierarchy.
  - `inspections`: `id`, `palm_id`, `image_url`, `status`, `created_at`.
  - `detections`: `id`, `inspection_id`, `class_name`, `confidence`, `bbox_x_min`, `bbox_y_min`, `bbox_x_max`, `bbox_y_max`.
  - `recommendations`: `id`, `inspection_id`, `response_text`, `model_name`, `citations_json`, `verification_status`.
  - `document_chunks`: `id`, `document_id`, `content`, `metadata_json`, `embedding` (vector(1536)).
  - `detection_corrections`: `id`, `inspection_id`, `corrected_by`, `original_bbox`, `corrected_bbox`, `created_at`.
- **Redis 7**:
  - Stream `stream:inspection_jobs`: Task queue for workers.
  - Key-value cache with TTL for session state and rate limit counters.
- **MinIO (Local S3-Compatible Object Store)**:
  - Bucket `ganodermascout-images`: Raw uploads and annotated thumbnail overlays.

### Security, Privacy & Reliability
- **Authentication & RBAC**:
  - Go Gateway validates JWT tokens signed by OAuth 2.0 provider.
  - Scopes: `scout:write` (upload inspections), `agronomist:review` (submit corrections), `admin:read` (view metrics & benchmarks).
- **Network Isolation**: All internal microservices (Vision, Agent, Postgres, Redis, Airflow) communicate inside an isolated internal Docker bridge network; only Go Gateway and Next.js frontend expose host ports.
- **Input Validation**: Strict MIME-type checking (magic byte inspection), image dimension bounds check, and request body size limits enforced at the gateway.
- **Job Reliability**: Redis stream consumer groups with manual acknowledgments; automatic retry with exponential backoff up to 3 attempts before dead-letter queue routing (`stream:inspection_dlq`).

---

## 5. Risks & Phased Roadmap

### Phased Rollout Plan

```
MVP (Portfolio Release)         v1.1 (Operational Hardening)       v2.0 (Field Operations)
- Single `docker compose up`    - Batch image upload per block     - Edge-device model quantization
- 4 Detectors evaluated         - Multi-user RBAC & SSO            - Offline sync PWA
- Dual-mode LLM (Cloud/Local)   - Automated Airflow corpus update  - Weather & soil data integration
- Benchmark & admin dashboards  - Advanced A/B prompt routing      - Active learning pipeline
```

#### Phase 1: MVP Core (Current Target)
- Complete single-command local deployment with Docker Compose (`docker compose up`).
- Functional multi-service pipeline: Go Gateway, Core API, Vision Inference with 4 models, LangGraph Agent with dual-mode LLM, Redis stream queue, and Next.js frontend.
- Bundled sample image dataset + pre-indexed agronomic knowledge base.
- Comprehensive benchmark dashboard presenting detection Pareto frontier and RAG comparative metrics.

#### Phase 2: v1.1 Operational Hardening
- Batch upload capability for entire plantation row scouts.
- Multi-user authentication with estate-level data partitioning.
- Automated weekly Airflow DAG to ingest newly published open-access agronomy literature.
- Automated regression test pipeline on every Git tag release.

#### Phase 3: v2.0 Plantation Deployment
- TensorRT / ONNX Runtime quantization for edge inference on low-power plantation hardware.
- Offline Progressive Web App (PWA) allowing scouts to queue uploads while out of cellular range.
- Environmental correlation pipeline combining rainfall, soil humidity, and BSR spread velocity.

### Technical & Project Risks and Mitigations

| Risk Factor | Probability | Impact | Mitigation Strategy |
| --- | --- | --- | --- |
| **Dataset IP / Paper Embargo** | High | High | Keep full dataset private via DVC; ship synthetic and approved sample datasets in repo so full stack runs out-of-the-box without violating publication rights. |
| **Out-of-Distribution Domain Shift** | High | High | Enforce site-disjoint evaluation; implement pre-inference image quality checks and explicit low-confidence escalation routes. |
| **Symptom Staging Misunderstanding** | Medium | High | Explicit disclaimers in UI that absence of conks does not equal a healthy palm, backed by retrieved agronomic citations explaining internal decay latency. |
| **LLM Hallucination of Chemicals** | High | Critical | Multi-agent verification barrier: strictly block and rewrite any response containing chemical dosages or brand names not found verbatim in retrieved sources. |
| **Service Sprawl Complexity** | Medium | Medium | Strict modular Docker Compose configuration; each service includes standalone unit/mock tests to enable independent local development. |
| **Local LLM Performance Bottlenecks** | Medium | Medium | Provide seamless dual-mode configuration: light cloud API mode for immediate quick testing vs. local Ollama/vLLM mode for air-gapped evaluation. |
