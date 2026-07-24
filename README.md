# CCTV AI Face Identity MVP

Single-camera face recognition and person tracking using existing AI-team
images in `test_images`, OpenCV YuNet/SFace, Ultralytics YOLO + BoT-SORT,
SQLite and optional direct NDI input.

The current runtime is the camera-local identity prototype. The complete,
measurable four-camera Milestone 1 contract is documented in
[`docs/MILESTONE_1.md`](docs/MILESTONE_1.md). In particular, visual proximity
is an **estimated interaction**, not proof that two people spoke.

## Pipeline

`test_images -> YuNet -> SFace embeddings -> SQLite -> NDI/webcam/video ->
YOLO + BoT-SORT -> full-frame YuNet -> tracked upper-body ROI fallback ->
original-pixel quality gate -> SFace -> identity voting -> overlay`

## Important data note

The current repository contains biometric images in `test_images`. Use them
only with the consent of each person and company authorization. The local
SQLite database, downloaded models, outputs and any new enrolment images are
excluded from Git.

## Setup

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\download_models.py
python scripts\init_db.py
```

## Delete previously captured local photos

This does not touch repository `test_images`.

```powershell
python scripts\clean_local_enrollment.py --yes
```

## Enrol all five AI-team folders

```powershell
python scripts\enroll_ai_team.py
```

The script accepts exactly one face per source image. It stores independent
photo embeddings first, then uses mild deterministic augmentation only when
the limited repository images require more vectors. CCTV JPEG-compression
variants are selected across different source photos before extra variants
from the same photo, so one image cannot consume the augmentation budget.
Each person is capped at the same configured number of stored vectors.

Do not copy distant or tiny faces directly into the enrollment folders.
Low-resolution samples are useful as a regression set, but unsafe gallery
templates can increase false identity matches. Evaluate a labelled ZIP while
automatically excluding exact copies of existing enrollment images:

```powershell
python scripts\evaluate_face_archive.py `
  "C:\Users\sarkar\Downloads\test_images.zip"
```

The evaluator never extracts or enrolls archive images. It validates member
paths and size limits, requires exactly one face, applies the original-pixel
live identity floor, and exits non-zero if any eligible face is assigned to
the wrong person. `TOO_SMALL` means the person can still be tracked, but the
camera did not provide enough facial pixels for a safe identity decision.

Check counts:

```powershell
@'
from src.config import Settings
from src.repository import FaceRepository

settings = Settings.from_env()
for name, person_id, count in FaceRepository(
    settings.database_path
).embedding_counts():
    print(name, person_id, count)
'@ | python
```

## Recognize one image

```powershell
python scripts\recognize_image.py "test_images\shrayan\Screenshot (19).png"
```

## Webcam

```powershell
python scripts\live_recognition.py --source 0
```

## Video file

```powershell
python scripts\live_recognition.py --source "videos\test.mp4"
```

A processed video is saved automatically under `outputs`.

Use the recording timeline rather than inference speed for event timestamps:

```powershell
python scripts\live_recognition.py `
  --source "videos\test.mp4" `
  --recording-start "2026-07-21T09:00:00+05:30" `
  --headless
```

For a CPU-only offline review, process every fifth source frame while
preserving source timestamps and output duration:

```powershell
python scripts\live_recognition.py `
  --source "videos\test.mp4" `
  --input-layout 2x2 `
  --frame-stride 5 `
  --headless
```

This is an offline review tradeoff, not a live-throughput claim. The tracker
sees fewer temporal samples, so use stride 1 for final tracking metrics.

Restrict reporting to one or more rectangular regions with normalized coordinates
`x1,y1,x2,y2`. Tracks outside the region are removed before face recognition,
identity prediction, overlays and event logging:

```powershell
python scripts\live_recognition.py `
  --ndi --ndi-source 0 --input-layout 2x2 `
  --restricted-roi 0.10,0.10,0.45,0.95 `
  --restricted-roi 0.55,0.10,0.90,0.95
```

For a 2x2 mosaic, the coordinates refer to the complete mosaic. This is a
center-point gate: a person is reported only while the center of their tracked
box is inside any restricted rectangle. Each active rectangle is drawn in
yellow on the live view.

For interactive selection, run without `--headless`, draw the first box on the
first frame, and press Enter or Space to confirm. PowerShell then asks whether
you want to draw another box:

```powershell
python scripts\live_recognition.py `
  --ndi --ndi-source 0 --input-layout 2x2 --select-roi
```

If `--recording-start` is omitted, durations still use video-frame time, but
the first frame receives a synthetic current-UTC epoch.

## Distant-camera handling

The live pipeline preserves normal full-frame face recognition for nearby
people. If a tracked person has no identity-quality associated face, it
retries only that person's padded upper-body region at up to 2x working
resolution and a lower YuNet score threshold. The bounding box and all five
landmarks are then mapped back to the original frame before SFace alignment.
Enlarged pixels never count as new identity detail.

Relevant `.env` controls:

```dotenv
FACE_MATCH_THRESHOLD=0.45
FACE_MATCH_MARGIN=0.05
FACE_DETECTOR_BACKEND=scrfd
FACE_DETECTION_SCORE_THRESHOLD=0.70
LIVE_FACE_DETECTION_MIN_SIZE=12
LIVE_FACE_IDENTITY_MIN_SIZE=32
LIVE_FACE_ROI_UPSCALE=2.0
LIVE_FACE_ROI_RATIO=0.55
LIVE_FACE_ROI_SCORE_THRESHOLD=0.60
LIVE_FACE_ROI_MAX_DIMENSION=640
YOLO_IMAGE_SIZE=640
```

The 32px value is a software safety floor, not a camera target. Configure
cameras so faces are at least 48px high and preferably 64px or more under
motion, pose and compression. Faces below the configured identity floor remain
tracked but cannot cast identity votes. `FACE_MATCH_MARGIN` also rejects a
match when the best and next-best people are too similar.

`LIVE_FACE_ROI_MAX_DIMENSION` bounds only the enlarged ROI working image.
Native camera pixels are never downscaled by this safeguard.

SCRFD is the default face detector through InsightFace, while SFace continues
to generate the existing 128-dimensional database embeddings. Set
`FACE_DETECTOR_BACKEND=yunet` to use the previous OpenCV detector if the
InsightFace model pack is unavailable.

`FACE_LOW_LIGHT_ENHANCEMENT=lime` applies a conservative LIME-style lift only
to detector input. Recognition embeddings are still extracted from the
original camera pixels.

READY face embeddings are fused over a bounded five-observation window for
each active track. YuNet confidence and the original source-pixel face size
weight the samples; the buffer is cleared when a track expires or a camera
recovery creates a new identity epoch. This reduces one-frame identity flicker
without allowing stale evidence to persist forever.

Use each camera's native stream rather than a 2x2 recorder mosaic. If YOLO is
also missing distant *people*, try `YOLO_IMAGE_SIZE=960` on GPU-capable
hardware; it is intentionally 640 by default because 960 is substantially
slower on CPU. Resolution changes cannot reconstruct facial detail that was
discarded by the source stream.

## Activity storage and QR portal foundation

Database initialization now also creates additive tables for observed visits,
completed proximity-interaction estimates, profile-photo metadata and hashed,
expiring portal credentials:

```powershell
python scripts\init_db.py
```

Issue an expiring credential and QR image for an enrolled `person_id`:

```powershell
python scripts\issue_portal_credential.py PERSON_UUID `
  --base-url "https://portal.example.com" `
  --ttl-hours 24
```

Run the local dependency-free portal demo:

```powershell
python scripts\run_portal.py --host 127.0.0.1 --port 8080
```

The QR bearer token is exchanged immediately for a short-lived, HttpOnly
session cookie. Portal responses are person-scoped, non-cacheable, and expose
neither database IDs nor profile-photo filesystem paths. For any non-local
deployment, put the server behind HTTPS and pass `--secure-cookie`.

The activity repository, explicit gate-event processor, proximity state
machine and model-agnostic cross-camera associator are implemented and tested.
The associator accepts completed, normalized body embeddings from an external
ReID model and fails closed when topology, timing or similarity is ambiguous.
The single-camera live script does **not yet** extract those embeddings or
project tracks into organization-wide visits. That integration requires the
multi-camera worker, site calibration and ReID-model phases defined in the
Milestone 1 specification.

The live worker also feeds confirmed same-camera identities into the
proximity state machine. Two people must remain within the normalized
footpoint-distance threshold for 3 seconds before an interaction estimate is
started; short detector gaps receive a 1-second grace period. Completed
estimates are written to `proximity_interactions` and logged as
`INTERACTION_STARTED/UPDATED/ENDED`. This estimates sustained proximity; it
does not prove that the people were speaking.

Unknown tracks are also evaluated visually using temporary IDs such as
`unknown_track_12`. Their estimates are shown and written to
`conversation_logs`, but are not inserted into the person-scoped portal table
until a stable identity exists.

Every live run also writes a separate readable `.log` file and a `.csv` event
file under `conversation_logs`. These contain camera, both person IDs, UTC
start/event times, duration and normalized distance.

Open the simple desktop viewer with:

```powershell
python scripts\conversation_log_gui.py
```

## Direct NDI input

Install `ndi-python` through `requirements.txt`. On Windows, ensure the NDI
runtime/SDK is installed and that the sender and receiver are reachable on the
same network.

List sources:

```powershell
python scripts\list_ndi_sources.py
```

Connect by source index:

```powershell
python scripts\live_recognition.py --ndi --ndi-source 0
```

Connect by part of the source name:

```powershell
python scripts\live_recognition.py --ndi --ndi-source "CCTV"
```

If one NDI source is a four-camera 2x2 mosaic, enable quadrant processing:

```dotenv
NDI_LAYOUT=2x2
```

```powershell
python scripts\live_recognition.py --ndi --ndi-source 0
```

You can override the setting for one run, or replay a recorded mosaic:

```powershell
python scripts\live_recognition.py --ndi --input-layout single
python scripts\live_recognition.py `
  --source "videos\four_camera_mosaic.mp4" `
  --input-layout 2x2
```

Quadrants are fixed as Q1 top-left, Q2 top-right, Q3 bottom-left and Q4
bottom-right. Each receives an independent persistent person tracker and the
full configured YOLO inference size; boxes are mapped back onto the original
mosaic for display and face processing. This approximately doubles linear
person size at model input compared with processing the complete 2x2 mosaic,
at the cost of four detector passes per source frame. Event records use a
separate `[Q1]` to `[Q4]` camera namespace.

Black, flat-blue and other nearly uniform Video Loss quadrants are suppressed
after a short debounce instead of consuming continuous YOLO inference. Their
pixels are still checked cheaply on every frame, so tracking resumes after two
consecutive meaningful frames. Every recovery uses a new internal identity
epoch, preventing stale pre-outage track and face state from attaching to a
new person.

Quadrant mode keeps four small YOLO model/tracker instances resident and runs
four detector passes per composite frame. Expect a short first-frame warm-up
and substantially higher CPU/GPU work than `single` mode. The NDI sender must
keep a fixed output resolution during a session; if it renegotiates, restart
the worker so old tracking coordinates cannot attach to the resized mosaic.

The receiver requests BGRX/BGRA frames from NDI, copies the SDK-owned frame,
converts it to OpenCV BGR, and frees the NDI video buffer immediately.

## Overlay

Confirmed tracks show:

- Full name
- Shortened global person UUID
- Local tracker ID (and recovery epoch when applicable)
- Face similarity score
- YOLO person-detection confidence

Unconfirmed or low-confidence tracks show `UNKNOWN`.

With `--show-faces`, white face boxes passed the original-pixel quality gate;
orange boxes were detected but rejected as too small or otherwise unusable.
Each person label explains the last face-stage result, such as `TOO SMALL`,
`UNKNOWN`, `MATCH`, `not detected`, or `detector error`.

An identity is confirmed only after the same person receives the configured
number of votes inside the recent vote window. Once confirmed, it remains
attached while the same camera-local tracker ID remains active.

## Tests

```powershell
pytest -q
python -m compileall src scripts tests
```

## Limitations

- Existing screenshots provide limited pose and lighting diversity.
- Augmented embeddings are not independent photos.
- The threshold must be calibrated using real CCTV/NDI footage.
- ROI upscaling helps locate a face but cannot restore missing identity detail.
- Native per-camera streams are required; a recorder mosaic reduces the pixel
  budget available to every face.
- New tracker IDs after long occlusion must be recognized again.
- No body ReID, unknown-person cross-camera handoff, anti-spoofing or
  production multi-camera orchestration yet.
- Presence is reportable as organization time only after every site entrance
  and exit is instrumented; otherwise it is observed first-to-last-sighting
  time.

## Demo overlay update

The live window now exposes the complete decision process:

- Gray box: `UNKNOWN`
- Amber box: `PENDING`, including current vote progress
- Green box: `CONFIRMED`
- Explicit `Face similarity` and `Person detection` labels
- Top panel with source, FPS, active tracks, enrolment counts, threshold and recognition interval
- One-time `TRACK_STARTED`, `IDENTITY_CONFIRMED` and `TRACK_ENDED` console events

Keyboard controls:

```text
Q or Esc  close
R         reload embeddings from SQLite
```

Recommended demo command:

```powershell
python scripts\live_recognition.py `
  --source 0 `
  --show-faces `
  --output "outputs\webcam_demo.mp4"
```

For NDI:

```powershell
python scripts\live_recognition.py `
  --ndi `
  --ndi-source 0 `
  --show-faces `
  --output "outputs\office_ndi_demo.mp4"
```
