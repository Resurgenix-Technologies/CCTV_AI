# Project Technology Stack & Architecture

This document provides a comprehensive technical breakdown of all libraries, frameworks, neural networks, algorithms, hardware acceleration, and infrastructure used to build this real-time CCTV multi-camera tracking, face recognition, body Re-ID, and conversation analytics system.

---

## 1. Core Language, Hardware & Runtime Acceleration

- **Python 3.10+**: Core programming language used for video stream ingestion, neural net inference, tracking algorithms, and concurrency.
- **PyTorch (CUDA GPU Acceleration)**: Machine learning runtime with native CUDA GPU support (`cuda:0` on **NVIDIA GeForce RTX 5070**).
- **FP16 Half-Precision Execution (`half=True`)**: Ultra-fast 16-bit floating point model inference for YOLO pose tracking.
- **Hardware Acceleration Module (`hardware_acceleration.py`)**:
  - Auto-detects GPU hardware capabilities and sets PyTorch CUDA device strings.
  - Automatically configures OpenCV DNN backends (CUDA/DirectML/OpenVINO) with CPU fallback protection.
- **NumPy**: High-performance vector math, matrix operations, L2 normalization, and cosine similarity calculation.

---

## 2. Computer Vision & Frame Processing

- **OpenCV (`opencv-python` & `opencv-contrib-python`)**:
  - Video stream capture and native-resolution frame processing.
  - Image enhancement algorithms:
    - **YCrCb CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Dynamic contrast enhancement for low-light CCTV streams.
    - **Bilateral Filtering**: Noise reduction while retaining sharp facial edges (eyes, nose, mouth).
    - **Unsharp Masking & Auto Gamma Correction**: Image brightness and edge contrast optimization.
    - **Laplacian Variance**: Real-time sharpness scoring and blur detection.
  - OpenCV DNN Module: ONNX model execution for YuNet & SFace/ArcFace engines.
- **MediaPipe (`mediapipe.solutions.face_mesh`)**:
  - 468-point 3D landmark mesh used to extract mouth corner and lip coordinates for Mouth Aspect Ratio (MAR) calculations.
- **Pillow (PIL)**:
  - Image decoding and EXIF orientation normalization for enrollment photos.

---

## 3. Deep Learning & Computer Vision Models

| Component | Model / Engine | Purpose | Output / Dimension |
| :--- | :--- | :--- | :--- |
| **Person Detection & Tracking** | **YOLO26 / YOLO11 / YOLOv8-Pose (Ultralytics)** | Multi-camera person detection, track ID assignment (ByteTrack/BoT-SORT), and body pose keypoint estimation via `load_yolo_pose_model()` (NVIDIA GPU FP16). | Bounding boxes, track IDs, 17-point pose keypoints |
| **Face Detection** | **YuNet (`face_detection_yunet_2023mar.onnx`)** | Ultra-lightweight face detection within zoomed body crops and 5-point facial keypoint extraction. | Bounding boxes, confidence score, 5 facial keypoints |
| **Face Recognition** | **SFace (`face_recognition_sface_2021dec.onnx`) & ArcFace** | Deep facial feature extraction and identity matching. | Dynamic 128-d & 512-d L2-normalized feature embeddings |
| **Super-Resolution** | **FSRCNN (`FSRCNN_x3.pb`)** | Deep learning-based 3x upscaling for small/distant face crops. | Upscaled BGR face region |

---

## 4. Pipeline & Custom Algorithmic Modules

### A. Native High-Res ROI & Quality Gating Pipeline ([face_pipeline.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/face_pipeline.py))
- **Native Resolution Mapping**: Maps bounding boxes directly to full-resolution native streams (`1080p`/`4K`) before crop enhancement, avoiding pre-downscaling resolution loss on small faces.
- **Multi-Stage Zooming**: Bounding-box zoom $\rightarrow$ Stage-1 enhancement $\rightarrow$ Face re-detection $\rightarrow$ Stage-2 zoom.
- **Landmark Pose Estimation**: Computes **Yaw Angle** and **Pitch Angle** using 5-point keypoint geometry.
- **Quality Gating**: Rejects side profiles ($|Yaw| > 38^\circ$), downward looks ($|Pitch| > 28^\circ$), low confidence ($< 0.75$), and blurry crops before embedding extraction.
- **Refined Match Decision Logic**: Enforces margin checks only on borderline matches ($< 0.48$), preventing high-confidence matches from being falsely marked as `Unknown`.

### B. Body Appearance Re-ID Continuity ([body_reid.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/body_reid.py))
- **Spatial Color Histogram Signatures**: Extracts LAB and HSV color histograms across upper, middle, and lower body zones.
- **Non-Face Identity Tracking**: When a person turns their face away or is obscured, body appearance cosine similarity ($\text{sim} \ge 0.72$) maintains identity continuity without track drops.

### C. Dynamic Temporal Embedding Fusion ([my_face_fusion.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/my_face_fusion.py))
- **Dynamic Dimension Support**: Supports 128-d and 512-d feature vectors dynamically.
- **Cosine Similarity Outlier Rejection**: Rejects new face embeddings with cosine similarity $< 0.35$ relative to the running track median vector.
- **Composite Weighting**: Combines face size, detection confidence, and image sharpness score.

### D. Track-Level Identity Voting ([identity_manager.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/identity_manager.py))
- **Rolling Window Majority Vote**: Deque window ($N=10$) with $K=6$ required matching votes to transition track status from `PENDING` $\rightarrow$ `CONFIRMED`.

### E. Multi-Modal Interaction & Handshake Analytics ([conversation_tracker.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/conversation_tracker.py))
- **Handshake Keypoint Detection**: Dynamic Euclidean distance evaluation between wrist keypoints (indices 9 & 10) of standing individuals ([gesture_tracker.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/gesture_tracker.py)).
- **Mouth Aspect Ratio (MAR) Variance**: Rolling variance of lip opening vs. width ([mouth_movement.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/mouth_movement.py)).
- **Gesture Activity Tracking**: Rolling variance of wrist-to-shoulder offset ([gesture_tracker.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/gesture_tracker.py)).
- **Adaptive Spatial Proximity**: Dynamic distance thresholding based on human height.
- **Session Lifecycle & Hysteresis**: 5-second activation threshold with a 4-second grace period.

---

## 5. Video Stream Ingestion & Infrastructure

- **NDI (Network Device Interface)**:
  - Low-latency IP camera streaming protocol via `NDIManager` ([ndi_handler.py](file:///c:/Users/inter/Downloads/PRIVATE_TEST_cctv-sarisht-test/ndi_handler.py)).
  - Supports multi-camera quad-view arrangement (`CAM1`, `CAM2`, `CAM3`, `CAM4`).
- **Concurrent Processing**:
  - `ThreadPoolExecutor` for parallel per-camera frame processing loops.
  - Thread-safe data stores using `threading.Lock`.

---

## 6. Storage, Data & Interfaces

- **CSV Logging**: Thread-safe structured log stores for conversations (`logs/conversations.csv`) and handshake events (`logs/handshakes.csv`).
- **SQLite Database**: Relational database storage for persistent identity registry and event history (`src/database.py`).
- **FastAPI / Uvicorn**: Web portal server for remote identity and credential management (`src/portal_server.py`).
- **Tkinter**: GUI desktop log viewer (`scripts/conversation_log_gui.py`).

---

## Summary Stack Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Video Ingestion Layer                           │
│           NDI Stream Ingestion  |  OpenCV Frame Capture            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Hardware Acceleration & Tracking                      │
│     YOLOv8x-Pose (NVIDIA RTX 5070 GPU FP16)  |  YuNet (Faces)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Quality Gating & Image Enhancement                     │
│  Pose Angle Estimation | Sharpness Scoring | CLAHE | Bilateral Filter  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Feature Extraction, Fusion & Re-ID                       │
│ 128-d/512-d ArcFace & SFace | Outlier Rejection | Body Re-ID Tracker │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Analytics & Presentation Layer                      │
│   Conversation Tracker (MAR + Gestures) | SQLite / CSV | OpenCV UI     │
└────────────────────────────────────────────────────────────────────────┘
```
