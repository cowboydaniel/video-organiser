# Roadmap: Automated Video Organiser

This roadmap outlines the work required to build a tool that watches a video, derives a concise title from audio and visual cues, and suggests an appropriate folder name for storage.

## Phase 1: Foundations and Ingestion
- Define core requirements and success criteria (accuracy thresholds, latency budgets, supported formats).
- Establish project structure (backend service, workers, storage, models, UI/CLI).
- Implement video ingestion pipeline: file upload/watch directory, metadata extraction (resolution, fps, duration), and checksum-based deduplication.
- Add basic logging, metrics, and feature-flag scaffolding.

## Phase 2: Audio and Visual Understanding
- Integrate speech-to-text transcription (e.g., Whisper) with language-specific handling and diarization.
- Implement scene and object detection (e.g., CLIP/ViT) to surface salient visual entities.
- Add temporal segmentation (shots/scenes) to align transcripts with visual cues.
- Build a multimodal summarization module that fuses transcript + visual tags into a concise title.
- Validate summarization against sample videos; iterate with prompt/parameter tuning.

## Phase 3: Folder Naming Strategy
- Define folder naming scheme (e.g., `{primary_subject}/{event_or_topic}/{date}`) with sanitization rules.
- Map extracted entities (people, locations, events) and timestamps into the naming template.
- Add configuration for user-specific preferences (e.g., date format, max length, stop-words).
- Implement fallback strategies when confidence is low (e.g., use date + hash).

## Phase 4: Orchestration and Reliability
- Introduce a task queue/worker model for long-running processing; ensure idempotency.
- Add caching for repeated frames/audio chunks and model outputs.
- Implement retries, circuit breakers for external ML services, and rate limiting.
- Provide health checks, structured logs, and dashboards for model performance (confidence, WER, relevance).

## Phase 5: Interfaces and UX
- CLI commands for batch processing and watch-mode (auto-process new files in a directory).
- Lightweight web UI to upload videos, preview generated titles, and accept/override folder suggestions.
- Notifications or webhooks when processing completes.
- Accessibility considerations (captions, keyboard navigation) and internationalization for titles.

## Phase 6: Data, Evaluation, and Privacy
- Create evaluation dataset with ground-truth titles and folder labels; automate regression tests.
- Track quality metrics (precision/recall on entities, title relevance, suggestion acceptance rate).
- Enforce privacy: local-only processing options, PII redaction in titles, and secure storage of artifacts.
- Document data retention and opt-out mechanisms.

## Phase 7: Deployment and Operations
- Provide containerized deployment with GPU support for heavy models.
- Offer configuration for offline/local models vs. cloud APIs.
- CI/CD pipeline with linting, tests, and reproducible builds; supply sample videos for demos.
- Rollout plan with feature flags and gradual exposure; incident playbooks for model drift or failures.

## Phase 8: Stretch Goals
- Active learning loop to improve models from user corrections.
- Multilingual title generation with automatic locale detection.
- Duplicate/near-duplicate detection across library to reduce clutter.
- Integrations with cloud storage providers and NAS devices.

## Deliverables Checklist
- [ ] Ingestion pipeline with metadata + deduplication.
- [ ] Speech and vision modules producing aligned transcripts and tags.
- [ ] Multimodal summarizer for candidate titles.
- [ ] Folder naming engine with configurable templates and fallbacks.
- [ ] CLI and UI for processing and reviewing suggestions.
- [ ] Observability and evaluation suite with quality metrics.
- [ ] Deployment artifacts (Dockerfile, CI workflows) and operational runbooks.
