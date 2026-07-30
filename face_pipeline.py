import os
from pathlib import Path
import numpy as np

from my_face_engine import (
    FaceEngine,
    FaceEngineError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    FaceTooSmallError,
    compute_face_sharpness,
)
from image_enhancement import (
    enhance_face_region,
    configure_superres,
    upscale_face_crop,
    generate_tta_variants,
    generate_cctv_domain_variants,
)
from mouth_movement import measure_mouth_aspect_ratio

# Configuration Parameters
FACE_CROP_UPSCALE = 3.5
FACE_CROP_MAX_DIMENSION = 900
FACE_DETECTION_SCORE_THRESHOLD = 0.7

MAX_YAW_ANGLE = 38.0
MAX_PITCH_ANGLE = 28.0
MIN_SHARPNESS_SCORE = 15.0

FACE_ZOOM_MARGIN_RATIO = 0.4
FACE_ZOOM_UPSCALE = 4.0
FACE_ZOOM_MAX_DIMENSION = 700

ENROLLMENT_MIN_FACE_SIZE = 25
RECOGNITION_MIN_FACE_SIZE = 18
RECOGNITION_MIN_SCORE = 0.75
DEBUG_RECOGNITION = True

RETRY_THRESHOLDS = [0.7, 0.5, 0.35]
TTA_ENABLED = False

MATCH_THRESHOLD = 0.40
MATCH_MIN_MARGIN = 0.06


def run_detection_with_threshold(image: np.ndarray, score_threshold: float, engine: FaceEngine):
    """Temporarily sets detection threshold on FaceEngine and performs detection."""
    previous_threshold = engine._detector.getScoreThreshold()
    engine._detector.setScoreThreshold(score_threshold)
    try:
        return engine.detect_faces(image)
    finally:
        engine._detector.setScoreThreshold(previous_threshold)


def detect_with_retries(image: np.ndarray, engine: FaceEngine):
    """Attempts face detection across multiple decreasing confidence thresholds."""
    for threshold in RETRY_THRESHOLDS:
        detections = run_detection_with_threshold(image, threshold, engine)
        if detections:
            return detections
    return []


def extract_robust_embedding(working_image: np.ndarray, initial_detection, engine: FaceEngine) -> np.ndarray | None:
    """Extracts normalized face embedding with optional test-time augmentation (TTA)."""
    embeddings: list[np.ndarray] = []

    try:
        base_result = engine.extract_embedding(working_image, initial_detection)
        embeddings.append(base_result.embedding)
    except FaceEngineError:
        pass

    if TTA_ENABLED:
        for variant in generate_tta_variants(working_image):
            try:
                variant_detections = detect_with_retries(variant, engine)
                if not variant_detections:
                    continue
                variant_result = engine.extract_embedding(variant, variant_detections[0])
                embeddings.append(variant_result.embedding)
            except FaceEngineError:
                continue

    if not embeddings:
        return None

    stacked = np.stack(embeddings, axis=0)
    averaged = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(averaged))
    if norm <= 1e-12:
        return None
    return (averaged / norm).astype(np.float32)


def zoom_and_recognize_face(
    full_res_frame: np.ndarray,
    body_box_native: tuple[int, int, int, int],
    engine: FaceEngine,
):
    """Multi-stage zooming, crop enhancement, face detection, embedding extraction, and mouth MAR measurement."""
    fx1, fy1, fx2, fy2 = body_box_native
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    body_crop = full_res_frame[fy1:fy2, fx1:fx2]
    if body_crop.size == 0:
        return None

    enhanced_body = enhance_face_region(body_crop)
    body_h, body_w = enhanced_body.shape[:2]

    stage1_w = int(body_w * FACE_CROP_UPSCALE)
    stage1_h = int(body_h * FACE_CROP_UPSCALE)
    longest = max(stage1_w, stage1_h)
    if longest > FACE_CROP_MAX_DIMENSION:
        shrink = FACE_CROP_MAX_DIMENSION / longest
        stage1_w = max(1, int(stage1_w * shrink))
        stage1_h = max(1, int(stage1_h * shrink))

    stage1_working = upscale_face_crop(enhanced_body, stage1_w, stage1_h)

    try:
        stage1_detections = detect_with_retries(stage1_working, engine)
    except FaceEngineError:
        return None

    if not stage1_detections:
        return None

    stage1_best = stage1_detections[0]

    scale_x1 = stage1_w / body_w
    scale_y1 = stage1_h / body_h

    face_x1, face_y1, face_x2, face_y2 = stage1_best.box
    native_face_x1 = int(face_x1 / scale_x1) + fx1
    native_face_y1 = int(face_y1 / scale_y1) + fy1
    native_face_x2 = int(face_x2 / scale_x1) + fx1
    native_face_y2 = int(face_y2 / scale_y1) + fy1

    face_w = native_face_x2 - native_face_x1
    face_h = native_face_y2 - native_face_y1
    if face_w <= 0 or face_h <= 0:
        return None

    margin_x = int(face_w * FACE_ZOOM_MARGIN_RATIO)
    margin_y = int(face_h * FACE_ZOOM_MARGIN_RATIO)

    full_h, full_w = full_res_frame.shape[:2]
    zx1 = max(0, native_face_x1 - margin_x)
    zy1 = max(0, native_face_y1 - margin_y)
    zx2 = min(full_w, native_face_x2 + margin_x)
    zy2 = min(full_h, native_face_y2 + margin_y)

    if zx2 <= zx1 or zy2 <= zy1:
        return None

    face_only_crop = full_res_frame[zy1:zy2, zx1:zx2]
    if face_only_crop.size == 0:
        return None

    enhanced_face = enhance_face_region(face_only_crop)
    crop_h, crop_w = enhanced_face.shape[:2]

    zoom_w = int(crop_w * FACE_ZOOM_UPSCALE)
    zoom_h = int(crop_h * FACE_ZOOM_UPSCALE)
    longest_zoom = max(zoom_w, zoom_h)
    if longest_zoom > FACE_ZOOM_MAX_DIMENSION:
        shrink = FACE_ZOOM_MAX_DIMENSION / longest_zoom
        zoom_w = max(1, int(zoom_w * shrink))
        zoom_h = max(1, int(zoom_h * shrink))

    zoomed_working = upscale_face_crop(enhanced_face, zoom_w, zoom_h)

    try:
        final_detections = detect_with_retries(zoomed_working, engine)
    except FaceEngineError:
        return None

    if not final_detections:
        return None

    final_best = final_detections[0]
    width_scale = zoom_w / crop_w
    height_scale = zoom_h / crop_h
    real_width = final_best.width / width_scale
    real_height = final_best.height / height_scale

    robust_embedding = extract_robust_embedding(zoomed_working, final_best, engine)
    if robust_embedding is None:
        return None

    mar = measure_mouth_aspect_ratio(zoomed_working)
    sharpness = compute_face_sharpness(zoomed_working)
    yaw = final_best.yaw_angle
    pitch = final_best.pitch_angle

    return robust_embedding, real_width, real_height, final_best.score, mar, sharpness, yaw, pitch


def is_valid_face_quality(
    real_width: float,
    real_height: float,
    score: float,
    yaw_angle: float,
    pitch_angle: float,
    sharpness: float,
    min_size: float = RECOGNITION_MIN_FACE_SIZE,
    min_score: float = RECOGNITION_MIN_SCORE,
) -> bool:
    """Validates face crop size, confidence score, pose orientation, and image sharpness."""
    if real_width < min_size or real_height < min_size:
        return False
    if score < min_score:
        return False
    if abs(yaw_angle) > MAX_YAW_ANGLE or abs(pitch_angle) > MAX_PITCH_ANGLE:
        return False
    if sharpness < MIN_SHARPNESS_SCORE:
        return False
    return True


def box_iou(box_a, box_b) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def deduplicate_overlapping_faces(detections, iou_threshold: float = 0.35):
    """Filters overlapping face detections based on IoU threshold."""
    if len(detections) <= 1:
        return detections

    sorted_detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept = []

    for detection in sorted_detections:
        overlaps_existing = any(
            box_iou(detection.box, kept_item.box) >= iou_threshold
            for kept_item in kept
        )
        if not overlaps_existing:
            kept.append(detection)

    return kept


def keep_dominant_face_only(detections):
    """Deduplicates overlapping faces and returns only the single largest detected face crop."""
    deduped = deduplicate_overlapping_faces(detections)

    if len(deduped) <= 1:
        return deduped

    by_area = sorted(deduped, key=lambda d: d.width * d.height, reverse=True)
    return [by_area[0]]


def enroll_photo(image: np.ndarray, engine: FaceEngine) -> np.ndarray:
    """Enrolls a single photo into an embedding vector after multi-pass enhancement and quality validation."""
    detections = detect_with_retries(image, engine)
    detections = keep_dominant_face_only(detections)

    working_image = image
    if not detections:
        enhanced_full = enhance_face_region(image)
        detections = detect_with_retries(enhanced_full, engine)
        detections = keep_dominant_face_only(detections)
        if detections:
            working_image = enhanced_full

    if not detections:
        raise NoFaceDetectedError("No face detected.")
    if len(detections) != 1:
        raise MultipleFacesDetectedError(
            f"Expected one face, detected {len(detections)}."
        )

    detection = detections[0]
    x1, y1, x2, y2 = detection.box
    margin_x = int((x2 - x1) * 0.5)
    margin_y = int((y2 - y1) * 0.5)
    h, w = working_image.shape[:2]
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(w, x2 + margin_x)
    cy2 = min(h, y2 + margin_y)

    face_crop = working_image[cy1:cy2, cx1:cx2]
    if face_crop.size == 0:
        raise NoFaceDetectedError("Face crop is empty.")

    enhanced_crop = enhance_face_region(face_crop)
    crop_h, crop_w = enhanced_crop.shape[:2]
    target_w = int(crop_w * FACE_CROP_UPSCALE)
    target_h = int(crop_h * FACE_CROP_UPSCALE)
    longest_side = max(target_w, target_h)
    if longest_side > FACE_CROP_MAX_DIMENSION:
        shrink = FACE_CROP_MAX_DIMENSION / longest_side
        target_w = max(1, int(target_w * shrink))
        target_h = max(1, int(target_h * shrink))

    working = upscale_face_crop(enhanced_crop, target_w, target_h)
    width_scale = target_w / crop_w
    height_scale = target_h / crop_h

    local_detections = detect_with_retries(working, engine)
    local_detections = keep_dominant_face_only(local_detections)

    if not local_detections:
        raise NoFaceDetectedError("Face lost after upscaling the crop.")
    if len(local_detections) != 1:
        raise MultipleFacesDetectedError(
            f"Expected one face after upscaling, detected {len(local_detections)}."
        )

    best = local_detections[0]
    original_width = best.width / width_scale
    original_height = best.height / height_scale

    if original_width < ENROLLMENT_MIN_FACE_SIZE or original_height < ENROLLMENT_MIN_FACE_SIZE:
        raise FaceTooSmallError(
            f"Face is {original_width:.0f}x{original_height:.0f}px; "
            f"minimum is {ENROLLMENT_MIN_FACE_SIZE}x{ENROLLMENT_MIN_FACE_SIZE}px."
        )

    robust_embedding = extract_robust_embedding(working, best, engine)
    if robust_embedding is None:
        raise NoFaceDetectedError("Could not extract a stable embedding after augmentation.")
    return robust_embedding


class FaceMatcher:
    """Loads enrolled face dataset and matches query face embeddings against known identities."""

    def __init__(
        self,
        engine: FaceEngine,
        known_faces_dir: Path,
        threshold: float = MATCH_THRESHOLD,
        min_margin: float = MATCH_MIN_MARGIN,
    ):
        self.engine = engine
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.min_margin = min_margin
        self.known_embeddings: list[np.ndarray] = []
        self.known_names: list[str] = []

        self.load_database()

    def load_database(self):
        self.known_embeddings.clear()
        self.known_names.clear()

        if not self.known_faces_dir.exists():
            print(f"Known faces folder not found: {self.known_faces_dir}")
            return

        for person_name in os.listdir(self.known_faces_dir):
            person_folder = self.known_faces_dir / person_name
            if not person_folder.is_dir():
                continue

            accepted = 0
            for imgname in os.listdir(person_folder):
                imgpath = person_folder / imgname
                try:
                    img = self.engine.read_image(imgpath)
                    embedding = enroll_photo(img, self.engine)
                    self.known_embeddings.append(embedding)
                    self.known_names.append(person_name)
                    accepted += 1

                    for variant in generate_cctv_domain_variants(img):
                        try:
                            variant_embedding = enroll_photo(variant, self.engine)
                            self.known_embeddings.append(variant_embedding)
                            self.known_names.append(person_name)
                            accepted += 1
                        except FaceEngineError:
                            pass
                except FaceEngineError as e:
                    print(f"SKIPPED | {imgpath.name} | {e}")

            print(f"{person_name}: {accepted} photos enrolled")

        print("Total enrolled embeddings:", len(self.known_embeddings))

    def find_match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Finds closest matching identity for target embedding vector."""
        best_per_person: dict[str, float] = {}
        for name, known_emb in zip(self.known_names, self.known_embeddings):
            similarity = float(np.dot(embedding, known_emb))
            if name not in best_per_person or similarity > best_per_person[name]:
                best_per_person[name] = similarity

        if not best_per_person:
            return "Unknown", -1.0

        ranked = sorted(best_per_person.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else -1.0

        if top_score < self.threshold:
            return "Unknown", top_score

        # Only enforce strict margin check when top match score is borderline (< 0.48)
        if top_score < 0.48 and (top_score - runner_up_score) < self.min_margin:
            return "Unknown", top_score

        return top_name, top_score
