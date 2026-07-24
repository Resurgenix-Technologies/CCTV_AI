# CCTV AI Surveillance System — Milestone 1 Specification

## 1. Purpose

Milestone 1 turns the current single-camera recognition prototype into a
four-camera pilot that can:

- detect visible people and maintain camera-local tracks;
- probabilistically link those tracks across configured cameras;
- associate a known person only through an explicit identity anchor;
- estimate site and room presence;
- estimate proximity-based interaction episodes;
- retain auditable events and derived timelines; and
- provide an authenticated personal activity portal.

This document defines the claim the product may make, the identity and event
semantics, the missing engineering work, and the acceptance gates for the
milestone.

## 2. Honest product claim

> Within a calibrated and instrumented site, the system detects visible people,
> maintains camera-local tracklets, probabilistically links tracklets across
> configured cameras during a visit, estimates site/room presence, and records
> proximity-interaction episodes with confidence and observable gaps. A durable
> person identity is assigned only through check-in, enrolment, or optional
> consented face evidence.

Milestone 1 must **not** claim that it:

- detects every person under all lighting, occlusion, camera, or crowding
  conditions;
- produces an infallible or permanent identity from appearance;
- continuously observes a person while they are in a camera blind spot;
- proves that nearby people spoke or communicated;
- measures total site presence if site entrances and exits are not
  instrumented; or
- is suitable as the sole evidence for disciplinary, employment, access, or
  safety decisions.

In user-facing language, use **estimated proximity interaction** or
**co-presence episode**, not **conversation detected**. Use **logical journey
with observed gaps**, not uninterrupted cross-camera tracking.

## 3. Current repository versus Milestone 1

### 3.1 Implemented now

The repository currently provides a useful single-camera proof of concept:

- YOLO11 person detection and ByteTrack camera-local tracking;
- webcam, video-file, and optional NDI ingestion;
- YuNet face detection and SFace enrolment/recognition;
- face-to-person-box association;
- repeated identity voting for one active local track;
- SQLite storage for people, face embeddings, and the significant events
  `TRACK_STARTED`, `IDENTITY_CONFIRMED`, and `TRACK_ENDED`;
- an additive activity store for explicit visits, completed proximity episodes,
  hashed portal credentials and person-scoped read models;
- a deterministic proximity-interaction state machine and gate-event projection
  bridge using capture timestamps;
- a model-agnostic, fail-closed cross-camera tracklet associator that accepts
  externally produced body embeddings and enforces topology, travel time,
  ambiguity, overlap and trusted-anchor constraints;
- a dependency-free QR exchange, personal dashboard and read-only portal API;
- command-line enrolment, recognition, live overlay, and event tooling; and
- unit tests for the existing repositories, association, identity state, and
  schema migration.

The raw ByteTrack integer is currently meaningful only within one tracker run.
The current confirmed identity remains attached to that local track; it is not
a cross-camera identity resolver.

### 3.2 Missing for Milestone 1

- Four simultaneous camera workers with isolated tracker state and health
  monitoring.
- Body/person ReID model execution, embedding extraction and quality-controlled
  tracklet summaries (the downstream association core is present).
- Camera topology, room/zone definitions, entry/exit lines, ground-plane
  calibration, and clock-health checks.
- Durable global-subject decision persistence, merge/split correction and
  multi-worker coordination around the in-memory association core.
- Explicit QR/check-in-to-camera anchoring.
- Automatic camera-to-visit projection, room-presence intervals, calibrated
  interaction zones, and group participant semantics.
- A concurrent multi-tenant event store and idempotent ingestion path.
- Production authentication, admin review, a concurrent API service and
  durable/distributed sessions beyond the local portal demo.
- Product-grade privacy controls, tenant isolation, retention, audit, backup,
  recovery, and operational monitoring.

SQLite remains appropriate for local development. A multi-camera,
multi-organization service should use PostgreSQL for product data.

## 4. Identity contract

Tracking, identity, and presence are different concepts and must use different
IDs.

| Identifier | Meaning | Lifetime and rules |
| --- | --- | --- |
| `camera_session_id` | One connection/run of one camera worker | New UUID after worker or stream restart |
| `tracklet_id` | One uninterrupted local trajectory in one camera session | UUID; the ByteTrack integer is stored only as local metadata |
| `subject_id` | Pseudonymous global hypothesis linking tracklets | Normally visit-scoped; may be merged, split, or corrected |
| `person_id` | Durable organization record for a known employee or visitor | Assigned only by an authorized identity anchor |
| `visit_id` | One site-presence episode | Starts at controlled entry and ends at controlled exit |
| `credential_id` | Revocable QR credential | Contains no PII and is not itself a visual-tracking identity |

An unknown person may have a `subject_id`, tracklets, a visit, and interactions
without ever receiving a `person_id`.

### 4.1 Identity anchors

A QR card identifies a credential, not the person visible in a camera. The
required controlled-entry workflow is:

1. An authorized operator assigns the credential to a person or a temporary
   visit record.
2. The credential is scanned at a configured entrance.
3. A gate camera observes a person crossing the corresponding entry line.
4. If exactly one eligible track is present in the configured time window, the
   visit is anchored to that subject.
5. Multiple candidates, missing video, or low-quality evidence produces an
   `AMBIGUOUS` review item; the system does not guess.

Optional face matching may corroborate an anchor only where collection and use
are authorized. A face decision must use multiple good-quality observations and
a threshold calibrated against representative site footage.

Body ReID is appearance matching, not durable human identification. It is most
credible within the same visit and can fail when clothing changes or two people
look alike.

## 5. Proposed architecture

```text
RTSP / NDI cameras
        |
        v
Per-camera edge workers
decode -> detect -> local MOT -> zones/ground position -> tracklet evidence
        |
        v
Durable event stream / queue
        |
        +--> Global identity resolver
        |      topology + travel time + body ReID + optional face + QR anchor
        |
        +--> Visit and room-presence engine
        |
        +--> Interaction estimator
        |
        v
PostgreSQL event log and read models <--> Redis live state
        |
        +--> Encrypted short-retention evidence in S3/MinIO
        |
        v
Authenticated API -> visitor portal and admin review
```

### 5.1 Camera workers

Each worker owns one detector/tracker state and emits:

- camera and session IDs;
- monotonic sequence number plus UTC capture time;
- local track ID and global `tracklet_id`;
- bounding box, detection confidence, and zone/ground position;
- quality-selected body embedding summaries;
- optional face evidence, gated by policy; and
- start, update, end, line-crossing, and health events.

Do not retain every decoded frame in the database. Publish observation batches
at the sampling rate needed for identity and interaction calculations, while
retaining significant transitions durably.

### 5.2 Cross-camera resolver

The resolver compares complete or mature tracklets, not arbitrary single
frames. Candidate matches are gated in this order:

1. permitted camera transition from the site topology;
2. plausible travel-time window;
3. no impossible simultaneous occupancy;
4. quality-weighted body ReID similarity;
5. optional multi-frame face evidence; and
6. QR/check-in anchor evidence.

Use a calibrated acceptance threshold and an ambiguity margin between the best
and second-best candidate. If the decision is not sufficiently clear, create a
new subject or a review item. Prefer a missed handoff over silently merging two
different people.

Every decision stores the candidate scores, evidence methods, thresholds,
model/config versions, and decision time. Merge, split, revoke, and manual
correction are append-only events so downstream visits and interactions can be
recomputed.

### 5.3 Suggested pilot components

- Existing Python/OpenCV/Ultralytics code for edge inference, extended with a
  benchmarked body-ReID model.
- Redis Streams or NATS JetStream as the durable four-camera pilot queue.
- Redis for short-lived online association and interaction state.
- PostgreSQL for concurrent, multi-tenant event and application data.
- S3-compatible storage for encrypted evidence assets.
- FastAPI for internal ingestion and user/admin APIs.
- A mobile-responsive web frontend.
- Docker Compose on an agreed on-premises GPU host for the pilot; distributed
  orchestration is a later scaling decision.

## 6. Presence semantics

`visit` and `room_presence` are separate projections.

- A visit begins on an accepted controlled site-entry event.
- It ends on an accepted controlled site-exit event.
- Room presence begins/ends on calibrated zone transitions and cross-camera
  handoffs.
- Small observation gaps may be bridged by a configured grace period, but the
  gap remains visible in provenance.
- An open visit displays `in progress`; it does not invent an exit time.

If every site entrance and exit is not covered, the dashboard must report
**observed duration from first to last sighting**, not total time inside the
organization.

All capture times are stored in UTC. Store both `occurred_at` and `ingested_at`
so queue delay is not mistaken for physical duration.

## 7. Interaction definition

For Milestone 1, an interaction episode is a deterministic estimate based on
visual proximity. Its site-specific parameters must be versioned.

A pair qualifies when:

- both subjects are in the same calibrated interaction zone;
- a wall, room boundary, or configured barrier does not separate them;
- estimated ground-plane distance is below the configured threshold, initially
  expected to be approximately 1.5–2 metres;
- the condition persists for the configured minimum dwell, initially expected
  to be 10 seconds; and
- evidence quality is adequate. Facing/orientation may increase confidence but
  does not prove speech.

Use hysteresis so distance jitter does not repeatedly open and close episodes.
A short missing-observation allowance, initially expected to be 3–5 seconds,
may bridge occlusion. Duration is the sum of qualifying evidence intervals, not
an unchecked difference between first and last timestamps.

When the minimum dwell is reached, the logical start may be backdated to the
first qualifying sample. End an episode at the last qualifying sample after the
gap allowance expires. Store confidence, observed duration, bridged duration,
rule version, and contributing tracklets.

Model groups as an `interaction_episode` with participant membership intervals.
Derive pairwise totals for the personal dashboard; do not infer that every pair
in a nearby group spoke to each other.

Actual speech attribution would require a separately authorized audio or
multimodal capability and is outside Milestone 1.

## 8. Event and data model

### 8.1 Core entities

- Tenant/site: `tenants`, `sites`, `zones`, `cameras`, `camera_sessions`.
- Identity: `people`, `subjects`, `subject_person_links`,
  `biometric_templates`, `qr_credentials`.
- Tracking: `tracklets`, short-retention `track_observation_batches`,
  `identity_associations`.
- Presence: `visits`, `room_presence`.
- Interaction: `interaction_episodes`, `interaction_participants`,
  `interaction_segments`.
- Governance: `events`, `media_assets`, `review_items`, `audit_log`.

### 8.2 Event envelope

Every event contains:

- globally unique `event_id` and `idempotency_key`;
- `tenant_id`, `site_id`, camera/session identifiers where applicable;
- `occurred_at` and `ingested_at` UTC timestamps;
- subject, person, visit, tracklet, and interaction references where applicable;
- confidence and status;
- model, configuration, calibration, and rule versions; and
- a structured payload plus an optional evidence reference.

Minimum event vocabulary:

- `TRACKLET_STARTED`, `TRACKLET_ENDED`
- `ZONE_ENTERED`, `ZONE_EXITED`
- `IDENTITY_ASSOCIATED`, `IDENTITY_REVOKED`, `SUBJECT_MERGED`, `SUBJECT_SPLIT`
- `VISIT_STARTED`, `VISIT_ENDED`, `VISIT_CORRECTED`
- `INTERACTION_STARTED`, `INTERACTION_ENDED`, `INTERACTION_CORRECTED`
- `CAMERA_ONLINE`, `CAMERA_OFFLINE`, `PIPELINE_STALLED`

The event log is append-only. Visits, interaction totals, and portal timelines
are rebuildable read models. Corrections never destroy the original evidence or
audit trail.

## 9. API surface

### 9.1 Internal ingestion and operations

- `POST /v1/ingest/tracklet-batches` — ordered, idempotent observation batches.
- `POST /v1/ingest/events` — significant edge events.
- `POST /v1/cameras/{camera_id}/heartbeat` — health and clock status.
- Admin endpoints for check-in/out, ambiguous-association review, identity
  merge/split, and correction.

### 9.2 Personal portal

- `POST /v1/portal/exchange` — exchange a QR secret for a short-lived session.
- `GET /v1/portal/me` — approved profile data and photo.
- `GET /v1/portal/me/visits` — entry, exit, duration, and confidence/status.
- `GET /v1/portal/me/interactions` — authorized counterpart labels and totals.
- `GET /v1/portal/me/timeline` — presence and interaction events.

Portal endpoints are scoped to the authenticated person and tenant. They must
not accept an arbitrary `person_id` as authority.

## 10. QR, privacy, and security

### 10.1 QR credential

- Encode at least 128 bits of cryptographically random opaque secret; do not
  encode a name, phone number, person UUID, or visit history.
- Store only a keyed hash of the secret where feasible.
- Support issuance, expiry, revocation, replacement, and access audit.
- Exchange the QR secret for a short-lived, narrowly scoped web session.
- Keep secrets out of application logs, analytics, referrer headers, and error
  reports; use `Cache-Control: no-store` for personal pages.
- Rate-limit exchange attempts and protect against enumeration and replay.
- A permanent QR is a bearer credential. For production personal-history access,
  require a PIN, OTP, or equivalent second factor. If the pilot omits this, the
  UI and risk register must state that possession of the card grants access.

The credential-to-person assignment and credential-to-camera-track anchor are
both explicit, auditable operations.

### 10.2 Privacy and access controls

- Complete a site- and jurisdiction-specific review for CCTV, biometric, visitor,
  and employee data before deployment.
- Provide clear notice, purpose limitation, access/correction procedures, and a
  deletion/retention process.
- Treat face photos and embeddings as sensitive biometric data; encrypt them,
  segregate access, and keep face association optional.
- Enforce tenant isolation, least-privilege RBAC, TLS/mTLS, encryption at rest,
  secret rotation, immutable access audit, and tested backup restoration.
- Retain evidence crops only briefly and retain derived events only as long as
  the approved purpose requires. Existing CCTV video follows the separately
  approved video-retention policy.
- Do not automatically reveal another participant's name. Use a name only when
  the organization's policy and applicable legal basis permit it; otherwise use
  an approved alias or anonymous label.
- Surface low-confidence and corrected records instead of presenting them as
  facts. Provide human review before consequential use.
- Evaluate errors across representative lighting, camera angles, occlusion,
  clothing similarity, and population groups.

## 11. Phased delivery

### Phase 0 — Scope, site survey, and ground truth

- Freeze the operating envelope: cameras, resolution/FPS, lighting expectations,
  eligible-person visibility, target hardware, entrances, exits, zones, and
  known blind spots.
- Configure topology, barriers, transition windows, and ground-plane calibration.
- Establish NTP/clock monitoring.
- Capture and independently annotate a consented evaluation set separated from
  tuning/training data. It should include every camera and camera pair, at least
  50 complete visits, 100 handoffs, 50 qualifying interaction episodes, and 50
  negative/passing/co-presence cases where practical.
- Approve identity, biometric, retention, and counterpart-disclosure policies.

### Phase 1 — Reliable four-camera local tracking

- Run isolated inference/tracker workers for four simultaneous feeds.
- Namespace all local IDs with camera sessions and emit idempotent batches.
- Add camera health, queue health, timestamps, model/config versions, and
  graceful reconnect.
- Retain the existing face vote as optional local evidence rather than treating
  it as global identity.

### Phase 2 — Global subjects and visits

- Add body-ReID tracklet summaries and camera-pair calibration.
- Implement topology/time/conflict gating and ambiguity handling.
- Add QR gate anchoring and manual review.
- Produce site visits, room-presence intervals, confidence, and visible gaps.
- Implement audited merge, split, revoke, and recomputation.

### Phase 3 — Interaction estimation

- Calibrate zones and ground distance.
- Implement the versioned dwell, distance, hysteresis, gap, barrier, and group
  rules.
- Produce evidence-backed episodes and pairwise portal aggregates.

### Phase 4 — Product backend and portal

- Introduce PostgreSQL, durable ingestion, read-model builders, retention jobs,
  and audit logging.
- Implement check-in/admin review APIs and the personal portal.
- Add secure QR exchange, session management, authorization, and disclosure
  policy enforcement.

### Phase 5 — Pilot hardening and sign-off

- Complete load, soak, outage/replay, backup/restore, deletion, model rollback,
  security, and end-to-end acceptance tests.
- Publish a pilot report with per-camera and per-camera-pair metrics, false
  merges/splits, uncertain cases, exclusions, and known limitations.

## 12. Measurable acceptance criteria

The following are proposed Milestone 1 gates. They may be adjusted during Phase
0 for the agreed site and hardware, but must be frozen before the held-out
acceptance set is evaluated. Thresholds must not be tuned on that acceptance
set. Report both aggregate and per-camera/per-camera-pair results.

### 12.1 Site readiness

- Every site ingress/egress is instrumented, or all relevant UI uses the phrase
  **observed duration**.
- Camera topology, room polygons, barriers, transition windows, and interaction
  zones have an approved configuration version.
- Camera clock skew remains below 250 ms during acceptance runs.
- Measured ground-position error is documented and is small enough relative to
  the chosen interaction-distance threshold.

### 12.2 Detection, local tracking, and ingestion

- Four agreed streams run at a minimum effective tracking rate of 10 FPS each
  on the target hardware with no growing queue during an 8-hour run.
- P95 capture-to-significant-event latency is below 2 seconds under normal
  operation.
- On eligible visible people in the held-out set: person-detection precision is
  at least 95%, recall at least 90%, and local tracking IDF1 at least 85%.
- Replaying an identical batch creates no duplicate durable event or read-model
  interval.
- A worker reconnect starts a new camera session and cannot collide with old
  local track IDs.

### 12.3 Cross-camera identity and presence

- Cross-camera association precision is at least 95%, recall at least 85%, and
  global IDF1 at least 85% on held-out routes.
- All deliberately ambiguous test cases are returned as unknown/review rather
  than confidently assigned to the wrong subject.
- QR anchoring succeeds in every unambiguous controlled single-person entry test
  and flags every deliberately ambiguous multi-candidate test.
- For fully instrumented visits, median absolute visit-duration error is at most
  5 seconds and P95 is at most 15 seconds.
- A merge/split correction rebuilds every affected visit, room interval,
  interaction, and portal total while preserving the original audit trail.

Any durable known-person face claim requires a separate representative
biometric evaluation and an approved false-match operating point. Passing the
body-ReID criteria alone does not satisfy that claim.

### 12.4 Interaction estimates

- The UI and API label results as estimated interactions and return confidence
  plus rule version.
- For annotated episodes of at least 30 seconds, interaction episode F1 is at
  least 0.85.
- Duration absolute error is at most 10 seconds or 10% of ground-truth duration,
  whichever is greater.
- Passing encounters, through-wall proximity, and mere same-room occupancy are
  represented in negative tests and do not create qualifying episodes at a rate
  greater than 5%.
- Replaying the same observations yields identical, non-overlapping episodes and
  totals.

### 12.5 Portal and security

- Profile, entry/exit, presence duration, interaction totals, and timeline
  reconcile with authorized backend projections to within one second of stored
  interval totals.
- A user can access only their tenant and authorized visit data; cross-tenant,
  IDOR, enumeration, expired-token, revoked-token, and replay tests pass.
- QR payload inspection reveals no PII or sequential identifier.
- Counterparty names follow the approved disclosure policy in every API and UI
  path.
- Personal-page responses are not cached, and secrets/biometric data do not
  appear in normal application logs.

### 12.6 Reliability and operations

- A continuous four-camera 8-hour soak run completes without process failure,
  unbounded memory growth, or increasing event lag.
- After a simulated 30-minute backend/network outage, buffered data replays with
  zero durable event loss and zero duplicate projections.
- Camera-offline and pipeline-stall alerts fire within 60 seconds.
- Backup restoration, approved-record deletion, credential revocation, and model
  rollback are demonstrated in the pilot environment.

## 13. Milestone completion definition

Milestone 1 is complete only when the four-camera held-out evaluation and the
security/reliability gates pass, the portal shows evidence-backed personal
timelines, and the pilot report documents limitations and uncertain cases.

A visually convincing overlay or a few successful face matches is a demo, not
Milestone 1 acceptance.
