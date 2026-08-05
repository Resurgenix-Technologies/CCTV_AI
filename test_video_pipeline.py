import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from hardware_acceleration import get_hardware_info, get_yolo_device_kwargs

hw_info = get_hardware_info()
_DEVICE = hw_info["device"]
_YOLO_KWARGS = get_yolo_device_kwargs()

print(f"Hardware Acceleration: {hw_info['gpu_name']} (Device: {_DEVICE}, FP16: {hw_info['use_fp16']})")

BASE_DIR = Path(__file__).resolve().parent

# Custom Modular Imports
from my_face_engine import FaceEngine
from my_face_fusion import FaceEmbeddingFusion
from identity_manager import IdentityManager, RecognitionResult
from conversation_logger import ConversationLogger
from image_enhancement import configure_superres
from mouth_movement import MouthMovementTracker
from gesture_tracker import GestureActivityTracker, extract_gesture_signal
from line_calibration import get_side
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
import db_integration


def process_video_file(
    input_video_path: Path,
    output_video_path: Path,
):
    print(f"Opening video file: {input_video_path}")
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {total_frames} frames @ {fps:.2f} FPS")

    # Output dimensions: 4 cameras in 2x2 grid (each processed display frame is 960x540)
    grid_w = 960 * 2
    grid_h = 540 * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (grid_w, grid_h))

    # Setup Models & Engines
    YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
    SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"

    primary_face_engine = FaceEngine(YUNET_MODEL, SFACE_MODEL, detection_score_threshold=0.85)

    SUPERRES_MODEL_PATH = BASE_DIR / "models" / "FSRCNN_x3.pb"
    if SUPERRES_MODEL_PATH.is_file():
        configure_superres(str(SUPERRES_MODEL_PATH), "fsrcnn", scale=3)

    known_faces_folder = BASE_DIR / "test_images"
    face_matcher = FaceMatcher(primary_face_engine, known_faces_folder)

    quadrant_names = ["CAM1", "CAM2", "CAM3", "CAM4"]
    camera_lines = {name: None for name in quadrant_names}  # Process full frames

    yolo_models = {name: YOLO("yolov8x-pose.pt") for name in quadrant_names}
    fusion_engines = {name: FaceEmbeddingFusion(max_samples=10) for name in quadrant_names}
    face_engines = {
        name: FaceEngine(YUNET_MODEL, SFACE_MODEL, detection_score_threshold=0.85)
        for name in quadrant_names
    }
    identity_managers = {
        name: IdentityManager(confirmation_votes=6, vote_window=10, ttl_frames=90)
        for name in quadrant_names
    }

    camera_state = {
        name: {"frame_count": 0, "cached_boxes": [], "known_track_ids": set()}
        for name in quadrant_names
    }

    conversation_log_csv = BASE_DIR / "logs" / "test_conversations.csv"
    conversation_logger = ConversationLogger(csv_path=conversation_log_csv)

    mouth_tracker = MouthMovementTracker(history_len=10, movement_std_threshold=0.018)
    gesture_tracker = GestureActivityTracker(history_len=8, movement_std_threshold=6.0)
    body_reid_tracker = BodyAppearanceReIDTracker(history_len=6, similarity_threshold=0.72)
    conversation_tracker = ConversationTracker()

    process_every_n_frames = 1
    RECOGNITION_INTERVAL_WHEN_CONFIRMED = 4
    last_recognition_frame: dict[tuple[str, int], int] = {}

    def process_camera_frame(cam_name: str, frame: np.ndarray) -> np.ndarray:
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
                            last_attempt = last_recognition_frame.get((cam_name, track_id), -999)
                            if (frame_index - last_attempt) < RECOGNITION_INTERVAL_WHEN_CONFIRMED:
                                should_run_recognition = False

                    if track_id != -1 and fx2 > fx1 and fy2 > fy1 and should_run_recognition:
                        last_recognition_frame[(cam_name, track_id)] = frame_index
                        zoom_result = zoom_and_recognize_face(
                            full_res_frame, (fx1, fy1, fx2, fy2), cam_face_engine
                        )

                        if zoom_result is not None:
                            embedding, real_width, real_height, det_score, mar, sharpness, yaw, pitch = zoom_result
                            mouth_tracker.update((cam_name, track_id), mar)

                            is_recognition_quality = is_valid_face_quality(
                                real_width, real_height, det_score, yaw, pitch, sharpness
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

                                recognition_result = RecognitionResult(
                                    matched=(matched_name != "Unknown"),
                                    person_id=matched_name if matched_name != "Unknown" else None,
                                    full_name=matched_name,
                                    similarity=score,
                                    embedding=fused_embedding,
                                )
                                identity_manager.observe(track_id, recognition_result, frame_index)

                                if fx2 > fx1 and fy2 > fy1:
                                    body_crop = full_res_frame[fy1:fy2, fx1:fx2]
                                    body_reid_tracker.update((cam_name, track_id), body_crop, matched_name)

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

                    if track_id != -1:
                        state_obj = identity_manager.get(track_id)
                        buffer_elapsed = (
                            (time.time() - state_obj.first_seen_time)
                            >= db_integration.RECOGNITION_BUFFER_SECONDS
                        )

                        if state_obj.confirmed and not state_obj.db_written:
                            if buffer_elapsed:
                                if state_obj.person_id:
                                    db_integration.add_track_id_to_ai_team(state_obj.full_name, cam_name, track_id)
                                else:
                                    face_id = db_integration.handle_unknown_visitor(cam_name, track_id, state_obj.best_embedding)
                                    state_obj.person_id = face_id
                                state_obj.db_written = True
                        elif (
                            not state_obj.confirmed
                            and not state_obj.db_written
                            and buffer_elapsed
                            and state_obj.best_embedding is not None
                        ):
                            face_id = db_integration.handle_unknown_visitor(cam_name, track_id, state_obj.best_embedding)
                            state_obj.person_id = face_id
                            state_obj.full_name = face_id
                            state_obj.confirmed = True
                            state_obj.db_written = True

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

            stale_ids = state["known_track_ids"] - current_track_ids
            if stale_ids:
                fusion.clear_many(stale_ids)
                for stale_id in stale_ids:
                    conversation_tracker.forget_track(cam_name, stale_id)
                    mouth_tracker.forget((cam_name, stale_id))
                    gesture_tracker.forget((cam_name, stale_id))
                    body_reid_tracker.forget((cam_name, stale_id))
                    last_recognition_frame.pop((cam_name, stale_id), None)
            state["known_track_ids"] = current_track_ids

            identity_manager.expire(frame_index)

            current_time = time.time()
            conversation_tracker.evaluate_pair_proximity(
                cam_name=cam_name,
                cached_boxes=state["cached_boxes"],
                identity_manager=identity_manager,
                mouth_tracker=mouth_tracker,
                gesture_tracker=gesture_tracker,
                current_time=current_time,
            )

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

    frame_num = 0
    start_time = time.time()

    while True:
        ret, full_frame = cap.read()
        if not ret:
            break

        frame_num += 1
        h, w = full_frame.shape[:2]
        half_h, half_w = h // 2, w // 2

        # Slice 4 quadrants
        quadrants = {
            "CAM1": full_frame[0:half_h, 0:half_w],
            "CAM2": full_frame[0:half_h, half_w:w],
            "CAM3": full_frame[half_h:h, 0:half_w],
            "CAM4": full_frame[half_h:h, half_w:w],
        }

        processed_frames = {}
        for cam_name in quadrant_names:
            processed_frames[cam_name] = process_camera_frame(cam_name, quadrants[cam_name])

        conversation_tracker.finalize_global_conversations(
            conversation_logger=conversation_logger,
            identity_managers=identity_managers,
        )

        for cam_name in quadrant_names:
            conversation_tracker.draw_talking_overlays(
                cam_name=cam_name,
                frame=processed_frames[cam_name],
                cached_boxes=camera_state[cam_name]["cached_boxes"],
                identity_manager=identity_managers[cam_name],
            )

        # Stitch into 2x2 grid (1920x1080)
        top_row = np.hstack((processed_frames["CAM1"], processed_frames["CAM2"]))
        bottom_row = np.hstack((processed_frames["CAM3"], processed_frames["CAM4"]))
        grid_frame = np.vstack((top_row, bottom_row))

        out_writer.write(grid_frame)

        if frame_num % 50 == 0 or frame_num == total_frames:
            elapsed = time.time() - start_time
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            print(f"Processed frame {frame_num}/{total_frames} ({fps_proc:.1f} FPS)")

    cap.release()
    out_writer.release()
    conversation_tracker.shutdown_active_conversations(
        conversation_logger=conversation_logger,
        identity_managers=identity_managers,
    )
    for identity_manager in identity_managers.values():
        identity_manager.close_all()

    print(f"\nProcessing complete! Output saved to: {output_video_path}")


if __name__ == "__main__":
    input_path = BASE_DIR / "Cropped version.mp4"
    output_path = BASE_DIR / "output.mp4"
    process_video_file(input_path, output_path)