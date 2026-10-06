# Product requirements document: GanodermaScout

- Version: 0.1 (draft) 
- Owner: Fazri Gading 
- Status: Ready for build planning 

## 1. Overview

GanodermaScout takes a photo of an oil palm base and a free-text question. It detects Ganoderma boninense fruiting bodies, labels each as a primordium or a mature basidiocarp, and returns a management recommendation that cites agronomy sources. The system keeps a history for each palm and plantation block, so it can describe how infection changes over time. An evaluation harness compares detection models, retrieval strategies, and memory on or off.

This is a portfolio project. It shows that one engineer can design, build, deploy, and evaluate a multi-service AI system on real domain data. It is a working MVP and demo, not a commercial release.

## 2. Goals and non-goals

### Goals

1. Deliver an end-to-end flow from photo upload to a cited recommendation, running locally with one `docker compose up`.
2. Show production patterns: an API gateway with OAuth, an async job queue with webhook callbacks, orchestrated data pipelines, experiment tracking, and monitoring.
3. Benchmark four detector families (YOLOv12, YOLOv13, RT-DETRv3, RF-DETR), three retrieval modes (dense, hybrid, hybrid with reranker), and memory on versus off, and publish the results on a benchmark page.
4. Produce a repository a reviewer can understand in ten minutes: architecture diagram, design doc, tradeoff notes, and a recorded demo.

### Non-goals

- Replacing a plantation pathologist or laboratory confirmation. The product is decision support.
- Early detection from the trunk or canopy. The MVP detects visible fruiting bodies only.
- Mobile apps, offline mode, and multi-language support.
- Drone or satellite imagery.
- Using every agent framework on the market. The project uses LangGraph and justifies the choice in the design doc.

## 3. Target users

| Persona | Need | Primary use |
| --- | --- | --- |
| Plantation field worker | Quick check of a palm and the next action | Upload photo, read result |
| Plantation agronomist | Review detections, see spread across blocks, get cited guidance | Review history, confirm or correct detections |
| Admin or ML engineer (also the hiring reviewer) | Inspect system health and model quality | Dashboards, benchmark page, experiment runs |

## 4. Scope

### In scope for the MVP

- Image upload with a text question, tagged with a block and palm ID.
- Detection and staging of fruiting bodies with bounding boxes.
- Retrieval over a curated corpus on Ganoderma and basal stem rot management, with citations.
- Per-palm and per-block memory of past detections.
- Alerts when detections cross a configured threshold.
- Evaluation harness and benchmark page.
- Admin dashboard for service health and model metrics.

### Out of scope for the MVP

- Multi-tenant organizations, billing, and payments.
- Real-time video.
- Automated ordering of treatments or chemicals.
- Disease types other than Ganoderma.

## 5. User stories and acceptance criteria

**US-1. Submit a palm photo.** As a field worker, I upload a photo and select a block and palm ID so that the palm is checked.

- Accepts JPEG and PNG up to 15 MB.
- Returns a job ID within 1 second.
- The UI shows job status (queued, processing, done, failed) without a page reload.
- The upload form shows capture guidance (distance, angle, lighting) taken from the project's own data collection protocol.

**US-2. See what was detected.** As a user, I see detections on the image.

- The result page draws bounding boxes labeled primordium or mature basidiocarp, each with a confidence score.
- The user can toggle the confidence threshold and see boxes appear or disappear.
- If image quality is too low (blur, glare, distance), the system asks for a new photo instead of reporting "no detection".

**US-3. Read a grounded recommendation.** As an agronomist, I read advice that cites sources.

- Each factual sentence links to at least one source passage.
- The answer states that absence of visible fruiting bodies does not mean the palm is healthy, because fruiting bodies are generally a late sign. The agent must back this statement with a retrieved source.
- Low confidence leads to a request for expert confirmation.

**US-4. Track progression.** As an agronomist, I see how a palm and its block change over time.

- A palm timeline shows each photo, its detections, and stage changes.
- A block view shows counts of affected palms by date.
- With memory on, the agent can mention a previous detection for the same palm in its answer.

**US-5. Receive alerts.** As an agronomist, I get notified about serious findings.

- A mature basidiocarp detection, or a block-level count above a configured threshold, triggers an n8n workflow that sends an email or Slack message within 60 seconds.

**US-6. Correct a detection.** As an agronomist, I mark a detection as wrong or add a missed one.

- Corrections are stored and exported as a candidate set for the next evaluation set version.

**US-7. Run an evaluation.** As an engineer, I compare configurations.

- I select a detector, retrieval mode, memory flag, and prompt version, run it against the fixed test set, and see metrics in a table and charts.
- Results are stored and comparable across runs.

**US-8. Safety check.** As a user, I do not receive unsafe advice.

- The verification agent blocks or rewrites recommendations with unsupported chemical dosages, banned substances, or claims without a source.

## 6. System architecture

### Services

| Service | Language | Responsibility |
| --- | --- | --- |
| API gateway | Go | OAuth 2.0, rate limiting, request routing, webhook dispatch |
| Core API | Python (FastAPI) | Jobs, palm and block records, history, orchestration entry point |
| Vision service | Python (FastAPI, PyTorch, OpenCV) | Image quality check and detection, exposed as a REST inference API with a selectable model |
| Agent service | Python (LangGraph) | Multi-agent workflow (see below) |
| Worker | Python | Consumes jobs from the queue and runs the pipeline |
| Ingestion pipeline | Apache Airflow | Document ingestion, chunking, embedding refresh, weather data pull |
| Automation | n8n | Alert workflows |
| Frontend | Next.js (React, TypeScript) | Upload and chat UI, detection viewer, palm and block timelines, admin and benchmark pages |

### Data stores

- Postgres with pgvector for relational data (blocks, palms, images, detections), document chunks, and embeddings.
- Redis for the job queue (Redis Streams) and caching.
- Object storage (MinIO locally, S3-compatible) for images.
- MLflow for experiment tracking and the model registry, and DVC for datasets and model files.

### Agent workflow (LangGraph)

1. **Router agent** classifies the request (new inspection, history question, general management question).
2. **Vision agent** calls the vision service and summarizes detections together with the user's question.
3. **Memory step** loads the palm's and block's recent detections when memory is enabled.
4. **Retrieval agent** runs the configured retrieval mode and returns passages with source IDs.
5. **Answer agent** drafts the recommendation using only retrieved passages and detection data.
6. **Verification agent** checks citations and safety rules, then approves, rewrites, or rejects.

### Request flow

1. The frontend sends the image, block, palm ID, and question to the Go gateway with an OAuth access token.
2. The gateway forwards the request to the Core API, which stores the image, creates a job, and pushes it to the queue.
3. A worker calls the vision service and the agent workflow.
4. The result is saved to Postgres, and the gateway sends a webhook callback. The frontend also receives it through server-sent events.
5. If the alert rule matches, the Core API calls an n8n webhook.

## 7. Functional requirements

### Vision service

- FR-V1. Serve a `/detect` endpoint that accepts an image and a model name, and returns boxes, classes (primordium, mature basidiocarp), and confidence scores.
- FR-V2. Run an image quality check (blur, exposure, apparent distance) and return a quality flag before detection.
- FR-V3. Load models from the MLflow registry. At least four are registered: YOLOv12, YOLOv13, RT-DETRv3, and RF-DETR.
- FR-V4. Report inference latency per request so the benchmark page can compare precision against speed.

### Retrieval and knowledge base

- FR-R1. Ingest open-access PDFs and HTML pages on Ganoderma and basal stem rot, with metadata (title, publisher, year, URL, license).
- FR-R2. Chunk, embed, and store the documents in pgvector.
- FR-R3. Support three retrieval modes selectable by configuration: dense only, hybrid (BM25 plus dense with rank fusion), and hybrid plus reranker.
- FR-R4. Return source IDs and passage offsets so the UI can link citations.

### Memory

- FR-M1. Store every completed inspection against a palm ID and block ID.
- FR-M2. Provide a per-request memory toggle, which the evaluation harness uses.

### Gateway and APIs

- FR-G1. Implement OAuth 2.0 (authorization code flow with PKCE for the frontend, client credentials for service callers).
- FR-G2. Rate limit per client.
- FR-G3. Document the REST API with OpenAPI and sign webhooks for job completion.
- FR-G4. Optional stretch: a read-only GraphQL endpoint for history queries.

### Evaluation harness

- FR-E1. Keep a versioned test set with labeled images (boxes and stages), questions, and reference answers.
- FR-E2. Run a configuration (detector, retrieval mode, memory flag, prompt version) over the test set and log results to MLflow.
- FR-E3. Compute the metrics in section 9.
- FR-E4. Show comparison tables and charts on the benchmark page, including a precision against latency plot for the detectors.
- FR-E5. Support a simple A/B mode that routes a share of live requests to a second configuration and logs both outcomes.

### Frontend

- FR-F1. Upload page with image preview, block and palm selectors, question box, and job status.
- FR-F2. Result page with an image viewer that draws boxes, a threshold slider, the recommendation, and clickable citations.
- FR-F3. Palm timeline and block summary pages.
- FR-F4. Admin page with service health, latency, queue depth, and registered model versions.
- FR-F5. Benchmark page described in FR-E4.

## 8. Data

### Image data

Use the project's own annotated images of Ganoderma fruiting bodies at oil palm bases, collected with the documented acquisition setup. Record the camera, lens, distance range, and lighting conditions in the dataset card. For field testing, hold out images from plots or sites that do not appear in training, because same-site splits overstate performance.

Before publishing anything, confirm that the data and trained weights may be released without conflicting with the planned paper. If needed, publish a small sample set and keep the full data private. The repo must still run end to end on the sample set.

### Text corpus

Collect open-access sources on Ganoderma and basal stem rot: research papers, plantation board guidance, and extension material. Aim for 100 to 300 documents. Keep source metadata and licenses for every document so each citation is verifiable.

### Evaluation set

- Labeled detection images with a site-disjoint split from training.
- At least 100 image-question pairs with hand-written reference answers and the supporting passages marked.
- A small adversarial subset for safety tests (dosage requests, prompt injection text in questions).
- Versioned with DVC.

## 9. Evaluation and success metrics

| Area | Metric | MVP target |
| --- | --- | --- |
| Detection | mAP@50, mAP@50:95, and per-class AP and recall (primordium and mature) on the site-disjoint set | Report all four models. Recall on mature basidiocarps is the headline, and the targets come from your paper's baseline |
| Detection speed | Latency per image on CPU and on one GPU | Publish the numbers with the precision-latency plot |
| Retrieval | Recall@5 and MRR against marked passages | Hybrid plus reranker beats dense only by a measured margin |
| Answer quality | Faithfulness (claims supported by retrieved passages) and citation precision | At least 0.90 citation precision |
| Safety | Share of unsafe test prompts blocked by the verification agent | At least 0.95 on the adversarial subset |
| Memory | Answer quality difference with memory on versus off on follow-up cases | Report the difference, including cases where memory hurts |
| System | p95 end-to-end latency for one image | Publish the number from a local run |
| System | Job success rate over a 500-request load test | At least 99 percent |

The LLM judge for faithfulness uses a fixed rubric. Check a sample of judged results by hand and report agreement.

## 10. Non-functional requirements

- **Deployability.** The full stack starts with Docker Compose. Kubernetes manifests (kind or minikube) are a stretch goal.
- **Observability.** Each service exposes Prometheus metrics. Grafana dashboards cover latency, error rate, queue depth, and detection confidence distribution. Requests carry a trace ID across services.
- **Security.** No secrets in the repo. Uploads are validated by type and size. Prompt-injection text in documents or questions must not change system instructions, and the test set includes such cases.
- **Reliability.** Jobs retry with backoff, failed jobs go to a dead-letter queue, and webhook delivery is idempotent.
- **Reproducibility.** Pinned dependencies, seeded training runs, and one documented command that rebuilds the benchmark results.
- **CI/CD.** GitHub Actions runs linting, unit tests, a small evaluation smoke test on the sample set, and image builds on each pull request.

## 11. Ethics and responsible use

- The UI states that results are decision support and shows confidence honestly.
- The system never reports a palm as healthy based on missing detections. It states the limits of the method.
- The system refuses to give treatment or chemical dosage numbers unless a retrieved source supplies them for this disease and crop, and it shows that source.
- Agronomist corrections are stored and feed the next evaluation set version.
- The repo documents known failure modes, such as other fungi that resemble fruiting bodies, partial occlusion by debris, and the bias of the data toward the sites where it was collected.

## 12. Milestones

| Phase | Deliverables | Exit check |
| --- | --- | --- |
| 1. Foundation | Vision service with one detector, Postgres with pgvector, basic dense RAG, simple UI, Docker Compose | A palm photo returns boxes and a cited answer locally |
| 2. Agents and services | LangGraph workflow, Redis queue, worker, Go gateway with OAuth, webhooks, palm and block records | The full async flow works and history is saved |
| 3. Data and MLOps | Four detectors in MLflow, Airflow ingestion, DVC, n8n alerts, evaluation harness, benchmark page | Detectors, retrieval modes, and memory on or off compared in tables |
| 4. Hardening | Prometheus and Grafana, load test, safety tests, README, design doc, demo video | All section 9 metrics reported |
| 5. Stretch | Kubernetes manifests, a cloud ML deployment, GraphQL history endpoint, BI dashboard export | Optional |

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dataset cannot be shared because of paper timing or site agreements | Repo cannot be reproduced | Release a sample set, document the full dataset, and keep the pipeline runnable on the sample |
| Detection performance drops on new sites | Weak demo and misleading claims | Site-disjoint evaluation, quality checks, and a low-confidence path |
| Fruiting bodies are a late sign, so the product looks less useful | Reviewer questions the value | State the limit in the UI and README, and position the system as support for sanitation and block-level tracking |
| Scope grows across too many services | Project never finishes | Ship phases in order and cut phase 5 before cutting the evaluation |
| LLM judge is biased or noisy | Unreliable benchmark | Fixed rubric, hand-checked sample, reported agreement |
| Corpus licensing problems | Cannot publish the repo | Use open-access sources only and record licenses |
| Hallucinated advice | Harm to users and credibility | Retrieval-only answers, citation checks, verification agent, adversarial tests |

## 14. Skills coverage map

| Skill area from the target list | Where it appears in this project |
| --- | --- |
| Backend and API standards (REST, Webhooks, OAuth) | Go gateway, Core API, webhook callbacks |
| Go as a secondary language | Gateway with auth and rate limiting |
| ETL and orchestration | Airflow ingestion and embedding refresh |
| Postgres and vector databases | pgvector with hybrid search |
| Transformers, Hugging Face, OpenCV | Vision service with transformer detectors (RT-DETRv3, RF-DETR) and image quality checks |
| LLM frameworks, multi-agent design, memory | LangGraph workflow, per-palm and per-block memory |
| RAG and hybrid search experiments | Evaluation harness and benchmark page |
| Docker, CI/CD, Kubernetes | Compose stack, GitHub Actions, optional manifests |
| MLflow, DVC, monitoring, A/B testing | Model registry, dataset versioning, Grafana, FR-E5 |
| Workflow automation | n8n alerts |
| Ethical AI and safety | Verification agent, refusal rules, documented failure modes |

## 15. Naming and repository

The project name is GanodermaScout. "Scout" is the plantation term for walking blocks to inspect palms. Service and repository names use the lowercase prefix `ganodermascout`, for example `ganodermascout-gateway`, `ganodermascout-vision`, `ganodermascout-agents`, `ganodermascout-eval`, and `ganodermascout-web`. Before publishing, confirm the GitHub name and a matching domain are free.

## 16. Open questions

1. How much of the dataset and which trained weights can be published before the paper is accepted?
2. Do the images include palm or block identifiers and GPS tags, or do they need to be assigned during setup?
3. Which LLM provider and embedding model will you use, and what is the budget for benchmark runs?
4. Is a hosted public demo needed, or are a local setup and a recorded video enough?
