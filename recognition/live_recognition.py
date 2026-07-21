"""
live_recognition.py

Live NDI recognition using BOTH modalities, kept independent:

    FACE  -> InsightFace (tiled detection + ArcFace embeddings)
    BODY  -> YOLOv8 person detection + OSNet (torchreid) embeddings

Face and body identities are matched against their own separate
registries (registry/<mode>/face, registry/<mode>/body) and drawn
separately. They are NOT fused into a single score, and identity is
NOT yet tied to a persistent track — both of those are the next
stages (knowledge base sections 18-22, priority 6/7). Expect frame-to-
frame flicker until track-level temporal stability is added.

Usage:
    python live_recognition.py --mode baseline
    python live_recognition.py --mode augmented
    python live_recognition.py --mode baseline --source-index 1
"""

import os
import json
import argparse
import cv2
import numpy as np
# import NDIlib as ndi
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# CONFIG
# =====================================================

FACE_THRESHOLD = 0.60

# Placeholder. Per knowledge base section 23, thresholds must be set
# experimentally from same-person / different-person similarity
# distributions, not guessed - tune this using evaluate_registry.py
# results before trusting this in anything real.
BODY_THRESHOLD = 0.55

DEVICE = "cuda"   # set to "cuda" if a GPU is available
INSIGHTFACE_CTX_ID = 0 if DEVICE == "cuda" else -1

YOLO_WEIGHTS = r"D:\files\CCTV_AI\yolo11n.pt"
PERSON_DETECT_CONF = 0.4
BODY_CROP_PADDING = 0.05   # keep in sync with augment_dataset.py / build_body_registry.py

REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = ""

# Tiling config for distant-face detection.
TILE_GRID = (2, 2)
TILE_OVERLAP = 0.20
IOU_DEDUPE_THRESH = 0.4


# =====================================================
# ARGS
# =====================================================

parser = argparse.ArgumentParser(description="Live NDI face + body recognition.")
parser.add_argument("--mode", choices=["baseline", "augmented"], default="baseline")
parser.add_argument("--persons", default=None)
parser.add_argument("--face-registry", default=None)
parser.add_argument("--body-registry", default=None)
parser.add_argument("--source-index", type=int, default=0)
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", default="output.mp4", help="Output video path")
args = parser.parse_args()

PERSON_FILE = args.persons or os.path.join("../data", f"persons_{args.mode}.json")
FACE_REGISTRY_DIR = args.face_registry or os.path.join("../registry", args.mode, "face")
BODY_REGISTRY_DIR = args.body_registry or os.path.join("../registry", args.mode, "body")


# =====================================================
# LOAD PERSON DATABASE (names)
# =====================================================

with open(PERSON_FILE, "r") as f:
    persons = json.load(f)

print(f"Loaded {len(persons)} registered identities from {PERSON_FILE}")


# =====================================================
# LOAD REGISTRIES (face + body kept separate)
# =====================================================

def load_registry(registry_dir):
    database = {}
    if not os.path.isdir(registry_dir):
        print(f"[WARN] registry dir not found: {registry_dir}")
        return database

    for fname in os.listdir(registry_dir):
        if not fname.endswith(".npy"):
            continue
        global_id = fname[:-4]
        arr = np.load(os.path.join(registry_dir, fname))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        database[global_id] = arr

    return database


face_database = load_registry(FACE_REGISTRY_DIR)
body_database = load_registry(BODY_REGISTRY_DIR)

print(f"Face registry: {len(face_database)} identities from {FACE_REGISTRY_DIR}")
print(f"Body registry: {len(body_database)} identities from {BODY_REGISTRY_DIR}")


def identify(query_embedding, database, threshold):
    """Compare against every stored reference embedding for every
    person, keep the single best match. Never averages templates."""
    best_score = -1.0
    best_id = None

    query = query_embedding.reshape(1, -1)

    for global_id, db_embeddings in database.items():
        scores = cosine_similarity(query, db_embeddings)[0]
        score = float(np.max(scores))

        if score > best_score:
            best_score = score
            best_id = global_id

    if best_id is None or best_score < threshold:
        return "UNKNOWN", best_score

    return best_id, best_score


def person_name(global_id):
    if global_id == "UNKNOWN":
        return "UNKNOWN"
    return persons.get(global_id, {}).get("name", global_id)


# =====================================================
# LOAD MODELS (once)
# =====================================================

print("Loading InsightFace...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640), det_thresh=0.4)

print("Loading person detector (YOLOv8)...")
person_detector = YOLO(YOLO_WEIGHTS)

print("Loading OSNet feature extractor...")
body_extractor = FeatureExtractor(
    model_name=REID_MODEL_NAME, model_path=REID_MODEL_PATH, device=DEVICE
)

print("Models ready.\n")


# =====================================================
# TILED / MULTI-SCALE FACE DETECTION
# =====================================================

def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _dedupe_faces(faces, iou_thresh=IOU_DEDUPE_THRESH):
    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    kept = []
    for face in faces:
        if not any(_iou(face.bbox, k.bbox) > iou_thresh for k in kept):
            kept.append(face)
    return kept


def detect_faces_tiled(app, frame, grid=TILE_GRID, overlap=TILE_OVERLAP):
    h, w = frame.shape[:2]
    cols, rows = grid
    tile_w, tile_h = w / cols, h / rows

    all_faces = list(app.get(frame))

    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w - tile_w * overlap))
            y1 = max(0, int(r * tile_h - tile_h * overlap))
            x2 = min(w, int((c + 1) * tile_w + tile_w * overlap))
            y2 = min(h, int((r + 1) * tile_h + tile_h * overlap))

            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            for face in app.get(tile):
                face.bbox = face.bbox + np.array([x1, y1, x1, y1])
                if hasattr(face, "kps") and face.kps is not None:
                    face.kps = face.kps + np.array([x1, y1])
                all_faces.append(face)

    return _dedupe_faces(all_faces)


# =====================================================
# PERSON DETECTION + BODY CROP (matches build_body_registry.py)
# =====================================================

def detect_person_boxes(frame):
    results = person_detector.predict(
        frame, classes=[0], conf=PERSON_DETECT_CONF, device=DEVICE, verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    h, w = frame.shape[:2]
    padded_boxes = []

    for x1, y1, x2, y2 in xyxy:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * BODY_CROP_PADDING
        x2 += bw * BODY_CROP_PADDING
        y1 -= bh * BODY_CROP_PADDING
        y2 += bh * BODY_CROP_PADDING

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 > x1 and y2 > y1:
            padded_boxes.append((x1, y1, x2, y2))

    return padded_boxes


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def extract_body_embedding(crop):
    feats = body_extractor([crop])
    embedding = feats[0]
    embedding = embedding.cpu().numpy() if hasattr(embedding, "cpu") else np.array(embedding)
    return l2_normalize(embedding)


def face_center_inside(face_bbox, person_box):
    fx1, fy1, fx2, fy2 = face_bbox
    cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= cx <= px2 and py1 <= cy <= py2


def draw_label(frame, x1, y1, x2, y2, text, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, text, (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )


# =====================================================
# NDI INITIALIZATION
# =====================================================

# if not ndi.initialize():
#     raise RuntimeError("NDI initialization failed")

# finder = ndi.find_create_v2()
# print("Searching for NDI sources...")
# ndi.find_wait_for_sources(finder, 5000)
# sources = ndi.find_get_current_sources(finder)

# if len(sources) == 0:
#     raise RuntimeError("No NDI sources found")

# print("\nAvailable Sources:")
# for i, src in enumerate(sources):
#     print(f"{i}: {src.ndi_name}")

# if args.source_index >= len(sources):
#     raise RuntimeError(f"--source-index {args.source_index} out of range")

# recv = ndi.recv_create_v3()
# ndi.recv_connect(recv, sources[args.source_index])
# print(f"\nConnected to: {sources[args.source_index].ndi_name}")

cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.input}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    args.output,
    fourcc,
    fps,
    (width, height)
)

print(f"Input : {args.input}")
print(f"Output: {args.output}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")


# =====================================================
# MAIN LOOP
# =====================================================

print("\nStarting recognition (face + body, kept independent)...")

# while True:
#     frame_type, video_frame, _, _ = ndi.recv_capture_v2(recv, 1000)

#     if frame_type != ndi.FRAME_TYPE_VIDEO:
#         continue

#     try:
#         frame = cv2.cvtColor(video_frame.data, cv2.COLOR_YUV2BGR_UYVY)
#     except Exception:
#         ndi.recv_free_video_v2(recv, video_frame)
#         continue

#     ndi.recv_free_video_v2(recv, video_frame)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------------------------------
    # FACE DETECTION + MATCHING
    # -------------------------------------------------
    faces = detect_faces_tiled(face_app, frame)

    face_results = []  # (bbox, global_id, name, score)
    for face in faces:
        global_id, score = identify(face.embedding, face_database, FACE_THRESHOLD)
        face_results.append((face.bbox, global_id, person_name(global_id), score))

    matched_face_keys = set()

    # -------------------------------------------------
    # PERSON DETECTION + BODY MATCHING
    # -------------------------------------------------
    for (x1, y1, x2, y2) in detect_person_boxes(frame):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        body_embedding = extract_body_embedding(crop)
        body_id, body_score = identify(body_embedding, body_database, BODY_THRESHOLD)
        body_name = person_name(body_id)

        color = (0, 255, 0) if body_id != "UNKNOWN" else (0, 0, 255)
        draw_label(frame, x1, y1, x2, y2, f"BODY {body_id} | {body_name} | {body_score:.2f}", color)

        # Any face whose center falls inside this person box gets drawn
        # too, labeled separately - this is grouping for display only,
        # NOT identity fusion.
        for (fbbox, fid, fname, fscore) in face_results:
            key = tuple(fbbox)
            if key in matched_face_keys:
                continue
            if face_center_inside(fbbox, (x1, y1, x2, y2)):
                matched_face_keys.add(key)
                fx1, fy1, fx2, fy2 = map(int, fbbox)
                fcolor = (0, 255, 0) if fid != "UNKNOWN" else (0, 0, 255)
                draw_label(frame, fx1, fy1, fx2, fy2, f"FACE {fid} | {fname} | {fscore:.2f}", fcolor)

    # Faces that weren't inside any detected person box (e.g. the
    # person detector missed that person this frame) still get drawn.
    for (fbbox, fid, fname, fscore) in face_results:
        if tuple(fbbox) in matched_face_keys:
            continue
        fx1, fy1, fx2, fy2 = map(int, fbbox)
        fcolor = (0, 255, 0) if fid != "UNKNOWN" else (0, 0, 255)
        draw_label(frame, fx1, fy1, fx2, fy2, f"FACE {fid} | {fname} | {fscore:.2f}", fcolor)

    writer.write(frame)
    cv2.imshow("NDI CCTV Recognition (face + body, independent)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# CLEANUP
# =====================================================

cv2.destroyAllWindows()
# ndi.recv_destroy(recv)
# ndi.destroy()
cap.release()
writer.release()

print("Stopped")