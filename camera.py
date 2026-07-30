import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO

from hardware_acceleration import get_hardware_info, get_yolo_device_kwargs

hw_info = get_hardware_info()
_DEVICE = hw_info["device"]
_YOLO_KWARGS = get_yolo_device_kwargs()

print(f"Hardware Acceleration: {hw_info['gpu_name']} (Device: {_DEVICE}, FP16: {hw_info['use_fp16']})")
if _DEVICE == "cpu":
    print(
        "TIP: Agar tumhare paas NVIDIA GPU hai, 'pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121' (ya apne CUDA version ke hisaab se) "
        "install karo - GPU pe ye poora system kaafi zyada fast chalega."
    )

BASE_DIR = Path(__file__).resolve().parent

# Custom Project Imports
from my_face_engine import FaceEngine
from my_face_fusion import FaceEmbeddingFusion
from identity_manager import IdentityManager, RecognitionResult
from conversation_logger import ConversationLogger
from image_enhancement import configure_superres
from mouth_movement import MouthMovementTracker

# Newly Modularized Components
from ndi_handler import NDIManager
from gesture_tracker import GestureActivityTracker, extract_gesture_signal
from line_calibration import get_side, calibrate_camera_lines
from body_reid import BodyAppearanceReIDTracker
from face_pipeline import (
    FaceMatcher,
    zoom_and_recognize_face,
    is_valid_face_quality,
    RECOGNITION_MIN_FACE_SIZE,
    RECOGNITION_MIN_SCORE,
    DEBUG_RECOGNITION,
)
from conversation_tracker import ConversationTracker

# ==================== Models & Face Engines Setup ====================

YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"

primary_face_engine = FaceEngine(
    YUNET_MODEL,
    SFACE_MODEL,
    detection_score_threshold=0.85,
)

SUPERRES_MODEL_PATH = BASE_DIR / "models" / "FSRCNN_x3.pb"
if Path(SUPERRES_MODEL_PATH).is_file():
    configure_superres(str(SUPERRES_MODEL_PATH), "fsrcnn", scale=3)
else:
    print("Super-res model file not found - using cubic resize instead.")

# Load Known Faces Database & Enrolled Embeddings
KNOWN_FACES_FOLDER = BASE_DIR / "test_images"
face_matcher = FaceMatcher(primary_face_engine, KNOWN_FACES_FOLDER)

# ==================== NDI Video Stream Setup ====================

quadrant_names = ["CAM1", "CAM2", "CAM3", "CAM4"]
CAMERA_SOURCE_NAMES = {
    "CAM1": "",
    "CAM2": "",
    "CAM3": "",
    "CAM4": "",
}

ndi_manager = NDIManager(
    quadrant_names=quadrant_names,
    camera_source_names=CAMERA_SOURCE_NAMES,
    ndi_mode="independent",
    search_timeout=8.0,
)

# ==================== Per-Camera Pipeline Instances ====================

yolo_models = {name: YOLO("yolov8x-pose.pt") for name in quadrant_names}
fusion_engines = {name: FaceEmbeddingFusion(max_samples=10) for name in quadrant_names}
face_engines = {
    name: FaceEngine(YUNET_MODEL, SFACE_MODEL, detection_score_threshold=0.85)
    for name in quadrant_names
}
identity_managers = {
    name: IdentityManager(
        confirmation_votes=6,
        vote_window=10,
        ttl_frames=90,
    )
    for name in quadrant_names
}

camera_state = {
    name: {"frame_count": 0, "cached_boxes": [], "known_track_ids": set()}
    for name in quadrant_names
}

# ==================== Logging & Trackers Initialization ====================

CONVERSATION_LOG_CSV = BASE_DIR / "logs" / "conversations.csv"
conversation_logger = ConversationLogger(csv_path=CONVERSATION_LOG_CSV)

mouth_tracker = MouthMovementTracker(history_len=10, movement_std_threshold=0.018)
gesture_tracker = GestureActivityTracker(history_len=8, movement_std_threshold=6.0)
body_reid_tracker = BodyAppearanceReIDTracker(history_len=6, similarity_threshold=0.72)
conversation_tracker = ConversationTracker()

# Interactive Line Calibration per camera feed
camera_lines = calibrate_camera_lines(quadrant_names, ndi_manager.get_camera_frame)

# Processing Parameters
process_every_n_frames = 1
RECOGNITION_INTERVAL_WHEN_CONFIRMED = 4
LAST_RECOGNITION_FRAME: dict[tuple[str, int], int] = {}

# ==================== Frame Processing Per Camera ====================

def process_camera(cam_name: str, frame: np.ndarray) -> np.ndarray:
    """Processes a single camera frame through YOLO tracking, face recognition, and proximity logic."""
    state = camera_state[cam_name]
    full_res_frame = frame
    full_h, full_w = full_res_frame.shape[:2]

    display_frame = cv2.resize(full_res_frame, (960, 540))
    scale_x = full_w / 960
    scale_y = full_h / 540

    frame = display_frame
    line = camera_lines.get(cam_name)
    yolo_model = yolo_models[cam_name]
    fusion = fusion_engines[cam_name]
    identity_manager = identity_managers[cam_name]
    cam_face_engine = face_engines[cam_name]

    state["frame_count"] += 1
    frame_index = state["frame_count"]

    if frame_index % process_every_n_frames == 0:
        state["cached_boxes"] = []
        current_track_ids = set()

        results = yolo_model.track(
            frame, classes=[0], persist=True, verbose=False, imgsz=640, **_YOLO_KWARGS
        )

        for r in results:
            has_keypoints = r.keypoints is not None
            for box_index, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(box.id[0]) if box.id is not None else -1

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if line is not None:
                    side = get_side(center_x, center_y, line[0][0], line[0][1], line[1][0], line[1][1])
                    if side <= 0:
                        continue

                if track_id != -1:
                    current_track_ids.add(track_id)

                if track_id != -1 and has_keypoints:
                    try:
                        kp_xy = r.keypoints.xy[box_index].cpu().numpy()
                        kp_conf = r.keypoints.conf[box_index].cpu().numpy()
                        gesture_signal = extract_gesture_signal(kp_xy, kp_conf)
                        gesture_tracker.update((cam_name, track_id), gesture_signal)
                    except (IndexError, AttributeError):
                        pass

                fx1 = max(0, int(x1 * scale_x))
                fy1 = max(0, int(y1 * scale_y))
                fx2 = min(full_w, int(x2 * scale_x))
                fy2 = min(full_h, int(y2 * scale_y))

                should_run_recognition = True
                if track_id != -1:
                    progress_status = identity_manager.vote_progress(track_id).status
                    if progress_status == "CONFIRMED":
                        last_attempt = LAST_RECOGNITION_FRAME.get((cam_name, track_id), -999)
                        if (frame_index - last_attempt) < RECOGNITION_INTERVAL_WHEN_CONFIRMED:
                            should_run_recognition = False

                if track_id != -1 and fx2 > fx1 and fy2 > fy1 and should_run_recognition:
                    LAST_RECOGNITION_FRAME[(cam_name, track_id)] = frame_index
                    zoom_result = zoom_and_recognize_face(
                        full_res_frame, (fx1, fy1, fx2, fy2), cam_face_engine
                    )

                    if zoom_result is not None:
                        embedding, real_width, real_height, det_score, mar, sharpness, yaw, pitch = zoom_result
                        mouth_tracker.update((cam_name, track_id), mar)

                        is_recognition_quality = is_valid_face_quality(
                            real_width, real_height, det_score, yaw, pitch, sharpness
                        )

                        if DEBUG_RECOGNITION:
                            print(
                                f"[{cam_name}] ID:{track_id} zoomed_face="
                                f"{real_width:.0f}x{real_height:.0f}px "
                                f"score={det_score:.2f} yaw={yaw:.1f}° pitch={pitch:.1f}° "
                                f"sharpness={sharpness:.1f} quality_ok={is_recognition_quality}"
                            )

                        if is_recognition_quality:
                            fused_embedding = fusion.update(
                                track_id,
                                embedding,
                                detection_score=det_score,
                                face_size=(real_width, real_height),
                                sharpness_score=sharpness,
                            )
                            matched_name, score = face_matcher.find_match(fused_embedding)

                            if DEBUG_RECOGNITION:
                                print(
                                    f"[{cam_name}] ID:{track_id} match={matched_name} "
                                    f"similarity={score:.3f}"
                                )

                            recognition_result = RecognitionResult(
                                matched=(matched_name != "Unknown"),
                                person_id=matched_name if matched_name != "Unknown" else None,
                                full_name=matched_name,
                                similarity=score,
                            )
                            identity_manager.observe(track_id, recognition_result, frame_index)

                            # Update Body Re-ID appearance signature
                            if fx2 > fx1 and fy2 > fy1:
                                body_crop = full_res_frame[fy1:fy2, fx1:fx2]
                                body_reid_tracker.update((cam_name, track_id), body_crop, matched_name)

                # Fallback Body Re-ID matching when face recognition is unavailable or obscured
                if track_id != -1 and fx2 > fx1 and fy2 > fy1:
                    body_crop = full_res_frame[fy1:fy2, fx1:fx2]
                    current_status = identity_manager.vote_progress(track_id).status
                    if current_status != "CONFIRMED":
                        reid_name, reid_sim = body_reid_tracker.match_appearance((cam_name, track_id))
                        if reid_name is not None and reid_sim >= 0.72:
                            reid_result = RecognitionResult(
                                matched=True,
                                person_id=reid_name,
                                full_name=reid_name,
                                similarity=reid_sim,
                            )
                            identity_manager.observe(track_id, reid_result, frame_index)
                            if DEBUG_RECOGNITION:
                                print(f"[{cam_name}] ID:{track_id} ReID match={reid_name} sim={reid_sim:.3f}")

                if track_id != -1:
                    is_stationary = conversation_tracker.update_movement_and_check_stationary(
                        cam_name, track_id, center_x, center_y
                    )
                else:
                    is_stationary = False

                box_height = y2 - y1
                state["cached_boxes"].append(
                    (x1, y1, x2, y2, track_id, center_x, center_y, box_height, is_stationary)
                )

        identity_manager.mark_seen(current_track_ids, frame_index)

        # Cleanup stale tracks
        stale_ids = state["known_track_ids"] - current_track_ids
        if stale_ids:
            fusion.clear_many(stale_ids)
            for stale_id in stale_ids:
                conversation_tracker.forget_track(cam_name, stale_id)
                mouth_tracker.forget((cam_name, stale_id))
                gesture_tracker.forget((cam_name, stale_id))
                body_reid_tracker.forget((cam_name, stale_id))
                LAST_RECOGNITION_FRAME.pop((cam_name, stale_id), None)
        state["known_track_ids"] = current_track_ids

        identity_manager.expire(frame_index)

        # Evaluate Pairwise Spatial & Engagement Proximity
        current_time = time.time()
        conversation_tracker.evaluate_pair_proximity(
            cam_name=cam_name,
            cached_boxes=state["cached_boxes"],
            identity_manager=identity_manager,
            mouth_tracker=mouth_tracker,
            gesture_tracker=gesture_tracker,
            current_time=current_time,
        )

    # Render Visual Annotations
    if line is not None:
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)

    for (x1, y1, x2, y2, track_id, cx, cy, _box_height, _stationary) in state["cached_boxes"]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if track_id != -1:
            progress = identity_manager.vote_progress(track_id)
            if progress.status == "CONFIRMED":
                label = progress.candidate_name
            elif progress.status == "PENDING":
                label = f"Verifying ({progress.votes}/{progress.required_votes})"
            else:
                label = "Unknown"
        else:
            label = "Unknown"

        cv2.putText(
            frame,
            f"ID:{track_id} {label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    return frame


# ==================== Main Processing Loop ====================

def main():
    print("Starting live processing. Press 'q' to quit.")
    camera_thread_pool = ThreadPoolExecutor(max_workers=len(quadrant_names))

    try:
        while True:
            frames_by_camera = {}
            skip_iteration = False

            for cam_name in quadrant_names:
                cam_frame = ndi_manager.get_camera_frame(cam_name)
                if cam_frame is None:
                    skip_iteration = True
                    break
                frames_by_camera[cam_name] = cam_frame

            if skip_iteration:
                time.sleep(0.01)
                continue

            futures = {
                camera_thread_pool.submit(process_camera, cam_name, frames_by_camera[cam_name]): cam_name
                for cam_name in quadrant_names
            }
            processed_frames = {}
            for future in as_completed(futures):
                cam_name = futures[future]
                processed_frames[cam_name] = future.result()

            conversation_tracker.finalize_global_conversations(
                conversation_logger=conversation_logger,
                identity_managers=identity_managers,
            )

            for cam_name in quadrant_names:
                frame = processed_frames[cam_name]
                conversation_tracker.draw_talking_overlays(
                    cam_name=cam_name,
                    frame=frame,
                    cached_boxes=camera_state[cam_name]["cached_boxes"],
                    identity_manager=identity_managers[cam_name],
                )
                cv2.imshow(cam_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Cleaning up resources...")
        camera_thread_pool.shutdown(wait=True)
        conversation_tracker.shutdown_active_conversations(
            conversation_logger=conversation_logger,
            identity_managers=identity_managers,
        )
        for identity_manager in identity_managers.values():
            identity_manager.close_all()

        ndi_manager.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()