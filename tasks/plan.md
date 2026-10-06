# Implementation Plan: GanodermaScout

## Overview
GanodermaScout is an end-to-end multi-service AI decision-support platform for oil palm plantations. It detects and stages *Ganoderma boninense* fruiting bodies (primordium vs. mature basidiocarp) from palm base photographs, tracks historical disease progression across palms and blocks, and generates verified agronomic management recommendations citing scientific literature via a multi-agent LangGraph workflow. The entire system is designed for single-command local reproducibility (`docker compose up`) featuring a Go API gateway, Python FastAPI microservices, Redis Streams job queue, PostgreSQL with pgvector, MLflow evaluation, n8n alerting, and a Next.js 15 frontend.

## Architecture Decisions
- **Microservices Topology**: Polyglot architecture separating concerns — Go for the high-performance API Gateway (auth, rate limiting, routing), Python for AI/ML inference and LangGraph workflows, and TypeScript (Next.js) for the interactive UI.
- **Asynchronous Task Processing**: High-resolution image analysis and multi-turn agent retrieval run asynchronously via Redis Streams and Python workers, with real-time client updates delivered via Server-Sent Events (SSE).
- **Dual-Mode LLM & Embedding Engine**: Seamless configuration switch between Cloud APIs (OpenAI/Anthropic/Gemini) and local self-hosted runtimes (Ollama/vLLM + local BAAI/bge embeddings) to allow zero-cost air-gapped evaluation.
- **Defense-in-Depth AI Guardrails**: Strict multi-agent separation where a dedicated Verification Agent audits drafted advice, blocking hallucinated chemical dosages and prompt injections before output delivery.
- **Reproducible Evaluation Harness**: Automated benchmarking across 4 detector architectures (YOLOv12, YOLOv13, RT-DETRv3, RF-DETR), 3 retrieval modes (Dense, Hybrid with RRF, Hybrid + Reranker), and memory states on a site-disjoint holdout dataset.

## External Tracker Note
Tasks are tracked in **GitHub Issues** on [fazrigading/GanodermaScout](https://github.com/fazrigading/GanodermaScout/issues) with label `ready-for-agent` and mirrored locally in [tasks/todo.md](file:///home/fazrigading/Projects/GanodermaScout/tasks/todo.md).

## Task List & Phases

### Phase 1: Project Foundation & Storage Infrastructure
- [#1 Task 1: Repository structure and core Docker Compose environment](https://github.com/fazrigading/GanodermaScout/issues/1)
- [#2 Task 2: Relational database schema and migrations](https://github.com/fazrigading/GanodermaScout/issues/2)
- [#3 Task 3: Seed data and sample datasets](https://github.com/fazrigading/GanodermaScout/issues/3)

#### Checkpoint: Foundation
- [ ] Database containers healthy with pgvector enabled
- [ ] Redis stream and MinIO bucket access validated
- [ ] Base migrations execute cleanly

---

### Phase 2: Vision Service & Model Serving
- [#4 Task 4: Image quality inspection filter](https://github.com/fazrigading/GanodermaScout/issues/4)
- [#5 Task 5: Vision inference service with multi-detector registry](https://github.com/fazrigading/GanodermaScout/issues/5)

#### Checkpoint: Vision Pipeline
- [ ] Image quality rejects blurred / underexposed test images
- [ ] `/detect` endpoint serves inference across registered detector models
- [ ] Bounding boxes and confidence scores format strictly to contract

---

### Phase 3: Knowledge Base & Hybrid RAG Engine
- [#6 Task 6: Agronomy document ingestion and chunking pipeline](https://github.com/fazrigading/GanodermaScout/issues/6)
- [#7 Task 7: Dual-mode embedding and pgvector hybrid retrieval with RRF and reranker](https://github.com/fazrigading/GanodermaScout/issues/7)

#### Checkpoint: Knowledge Retrieval
- [ ] Markdown/PDF agronomy corpus chunked with metadata
- [ ] Hybrid dense + sparse vector search returns top-k cited passages
- [ ] Reranker improves MRR@5 against test queries

---

### Phase 4: LangGraph Multi-Agent Workflow
- [#8 Task 8: LangGraph stateful agent graph](https://github.com/fazrigading/GanodermaScout/issues/8)
- [#9 Task 9: Verification agent guardrails](https://github.com/fazrigading/GanodermaScout/issues/9)

#### Checkpoint: Multi-Agent Synthesis
- [ ] Agent state graph executes end-to-end from user query to recommendation
- [ ] Verification agent successfully blocks ungrounded chemical dosage claims
- [ ] Memory toggle correctly incorporates previous palm inspection records

---

### Phase 5: Core API & Asynchronous Job Queue
- [#10 Task 10: Asynchronous inspection worker and Redis Streams queue](https://github.com/fazrigading/GanodermaScout/issues/10)
- [#11 Task 11: Core API endpoints, MinIO image upload, and Server-Sent Events (SSE) stream](https://github.com/fazrigading/GanodermaScout/issues/11)
- [#12 Task 12: Palm & block historical progression timeline API with agronomist correction endpoints](https://github.com/fazrigading/GanodermaScout/issues/12)

#### Checkpoint: Core Backend & Jobs
- [ ] Inspection job dispatched to Redis Stream and consumed by worker
- [ ] Client receives SSE status updates from queued to completed
- [ ] Palm history and agronomist corrections persisted and queryable

---

### Phase 6: Gateway & Automation Integration
- [#13 Task 13: Go API Gateway](https://github.com/fazrigading/GanodermaScout/issues/13)
- [#14 Task 14: n8n event automation workflows for critical infection alerts](https://github.com/fazrigading/GanodermaScout/issues/14)

#### Checkpoint: Gateway & Alerting
- [ ] Go Gateway authenticates requests and enforces token-bucket rate limits
- [ ] Webhook triggers n8n alert workflow on mature basidiocarp detection

---

### Phase 7: Evaluation Harness & Benchmark Suite
- [#15 Task 15: Evaluation harness for detectors, RAG modes, and memory toggle](https://github.com/fazrigading/GanodermaScout/issues/15)
- [#16 Task 16: Benchmark metrics reporter and Pareto frontier generator](https://github.com/fazrigading/GanodermaScout/issues/16)

#### Checkpoint: Benchmarking
- [ ] Automated eval suite runs against site-disjoint holdout set
- [ ] Metrics (mAP@50, Recall@50, Faithfulness, Citation Precision) logged to MLflow
- [ ] Pareto frontier visualization generated

---

### Phase 8: Frontend Application
- [#17 Task 17: Next.js upload interface with real-time SSE progress and image quality preview](https://github.com/fazrigading/GanodermaScout/issues/17)
- [#18 Task 18: Interactive detection viewer with bounding box canvas, confidence slider, and cited advice](https://github.com/fazrigading/GanodermaScout/issues/18)
- [#19 Task 19: Palm/block timeline history view and benchmark dashboard](https://github.com/fazrigading/GanodermaScout/issues/19)

#### Checkpoint: Frontend UI
- [ ] Scout upload flow works seamlessly with live progress
- [ ] Bounding boxes render on canvas with reactive threshold slider
- [ ] Timeline and benchmark dashboards render without hydration errors

---

### Phase 9: Observability, Packaging & E2E Verification
- [#20 Task 20: Observability stack](https://github.com/fazrigading/GanodermaScout/issues/20)
- [#21 Task 21: Single-command `docker compose up` orchestration and end-to-end smoke test suite](https://github.com/fazrigading/GanodermaScout/issues/21)

#### Checkpoint: Final Release Readiness
- [ ] All containers boot healthy on `docker compose up`
- [ ] End-to-end smoke test script runs 100% green
- [ ] Grafana dashboards display live service metrics

---

## Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
| --- | --- | --- |
| **Dataset IP / Paper Embargo** | High | Include high-quality sample data in repo; isolate full research dataset behind DVC remote storage. |
| **Chemical Dosage Hallucinations** | Critical | Multi-agent verification barrier with deterministic regex and passage-matching; refuse numerical dosages not in citations. |
| **GPU vs. CPU Portability** | Medium | Auto-detect CUDA; fallback to lightweight PyTorch CPU inference and quantized models with warning in logs. |
| **Service Dependency Ordering** | Medium | Docker healthchecks on Postgres, Redis, and MinIO with `depends_on: condition: service_healthy`. |
| **Local LLM Availability** | Low | Dual-mode configuration supporting cloud OpenAI/Anthropic keys or local Ollama endpoints via env vars. |

## Open Questions
- None. Discovery resolved LLM dual-mode support, single-command local deployment target, and sample dataset policy.
