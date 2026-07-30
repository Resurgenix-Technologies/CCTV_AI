"""YuNet face detection, SFace alignment and embedding generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from hardware_acceleration import configure_opencv_dnn


def compute_face_sharpness(image_crop: np.ndarray) -> float:
    """Computes sharpness of a face crop using Laplacian variance."""
    if image_crop is None or image_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


EMBEDDING_DIMENSION: Final[int] = 128
SUPPORTED_IMAGE_EXTENSIONS: Final[set[str]] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


class FaceEngineError(RuntimeError):
    """Base face-processing exception."""


class MissingModelError(FaceEngineError):
    """Raised when a required ONNX model is unavailable."""


class InvalidImageError(FaceEngineError):
    """Raised when an image cannot be decoded."""


class NoFaceDetectedError(FaceEngineError):
    """Raised when no face is detected."""


class MultipleFacesDetectedError(FaceEngineError):
    """Raised when an enrolment image has multiple faces."""


class FaceTooSmallError(FaceEngineError):
    """Raised when a detected face is too small."""


class EmbeddingGenerationError(FaceEngineError):
    """Raised when SFace cannot generate a valid vector."""


@dataclass(frozen=True)
class FaceDetection:
    """A YuNet face detection."""

    box: tuple[int, int, int, int]
    score: float
    raw: np.ndarray

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def right_eye(self) -> tuple[float, float]:
        if self.raw.size >= 6:
            return (float(self.raw[4]), float(self.raw[5]))
        return (0.0, 0.0)

    @property
    def left_eye(self) -> tuple[float, float]:
        if self.raw.size >= 8:
            return (float(self.raw[6]), float(self.raw[7]))
        return (0.0, 0.0)

    @property
    def nose(self) -> tuple[float, float]:
        if self.raw.size >= 10:
            return (float(self.raw[8]), float(self.raw[9]))
        return (0.0, 0.0)

    @property
    def yaw_angle(self) -> float:
        """Rough estimation of face yaw angle in degrees (-90 to +90).
        0 = direct frontal face, negative = turned left, positive = turned right."""
        rx, ry = self.right_eye
        lx, ly = self.left_eye
        nx, ny = self.nose
        eye_dist = math.hypot(lx - rx, ly - ry)
        if eye_dist < 1e-5:
            return 0.0
        eye_mid_x = (rx + lx) / 2.0
        offset = (nx - eye_mid_x) / (eye_dist / 2.0)
        return float(np.clip(offset * 45.0, -90.0, 90.0))

    @property
    def pitch_angle(self) -> float:
        """Rough estimation of face pitch angle in degrees (-90 to +90)."""
        rx, ry = self.right_eye
        lx, ly = self.left_eye
        nx, ny = self.nose
        eye_dist = math.hypot(lx - rx, ly - ry)
        if eye_dist < 1e-5:
            return 0.0
        eye_mid_y = (ry + ly) / 2.0
        offset = (ny - eye_mid_y) / eye_dist
        return float(np.clip((offset - 0.40) * 60.0, -90.0, 90.0))


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """One detected face and its normalized embedding."""

    detection: FaceDetection
    aligned_face: np.ndarray
    embedding: np.ndarray


class FaceEngine:
    """Reusable YuNet/SFace engine."""

    def __init__(
        self,
        detector_model: Path,
        recognizer_model: Path,
        *,
        detection_score_threshold: float = 0.85,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self.detector_model = detector_model.resolve()
        self.recognizer_model = recognizer_model.resolve()

        self._validate_model(self.detector_model, "YuNet")
        self._validate_model(self.recognizer_model, "SFace")

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_model),
                "",
                (320, 320),
                detection_score_threshold,
                nms_threshold,
                top_k,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognizer_model), ""
            )
            configure_opencv_dnn(self._detector)
            configure_opencv_dnn(self._recognizer)
        except AttributeError as exc:
            raise FaceEngineError(
                "OpenCV lacks FaceDetectorYN/FaceRecognizerSF. "
                "Install opencv-contrib-python."
            ) from exc
        except cv2.error as exc:
            raise MissingModelError(
                "OpenCV could not load the ONNX models. "
                "Confirm the files are real ONNX binaries."
            ) from exc

    @staticmethod
    def _validate_model(path: Path, label: str) -> None:
        if not path.is_file():
            raise MissingModelError(f"Missing {label} model: {path}")
        if path.stat().st_size < 1024:
            raise MissingModelError(
                f"Invalid {label} model: {path}. The file is too small."
            )

    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        if not path.is_file():
            raise InvalidImageError(f"Image not found: {path}")
        try:
            from PIL import Image, ImageOps
            pil_image = Image.open(str(path))
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except (OSError, ValueError) as exc:
            raise InvalidImageError(f"Could not read image: {path}") from exc
        if image is None or image.size == 0:
            raise InvalidImageError(f"Could not decode image: {path}")
        return image

    @staticmethod
    def write_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower() or ".jpg"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise InvalidImageError(f"Could not encode image: {path}")
        encoded.tofile(str(path))

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError("Image is empty or invalid.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageError(f"Unsupported image shape: {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        prepared = self._prepare(image)
        height, width = prepared.shape[:2]
        self._detector.setInputSize((width, height))

        try:
            _, faces = self._detector.detect(prepared)
        except cv2.error as exc:
            raise FaceEngineError("YuNet failed to process the frame.") from exc

        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in np.asarray(faces, dtype=np.float32):
            if row.size < 15:
                continue
            x, y, w, h = (float(value) for value in row[:4])
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                FaceDetection(
                    box=(x1, y1, x2, y2),
                    score=float(row[14]),
                    raw=np.ascontiguousarray(row.copy()),
                )
            )

        return sorted(detections, key=lambda item: item.score, reverse=True)

    def detect_faces_in_region(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int],
        *,
        upscale: float = 1.0,
        score_threshold: float | None = None,
        max_dimension: int | None = None,
    ) -> list[FaceDetection]:
        """
        Detect inside a frame region and return frame-coordinate detections.
        """

        scale = float(upscale)
        if not np.isfinite(scale) or scale < 1.0:
            raise ValueError("upscale must be a finite value of at least 1.0.")
        if score_threshold is not None and not (
            0.0 <= score_threshold <= 1.0
        ):
            raise ValueError("score_threshold must be between 0 and 1.")
        if max_dimension is not None and max_dimension < 32:
            raise ValueError("max_dimension must be at least 32.")

        prepared = self._prepare(image)
        frame_height, frame_width = prepared.shape[:2]
        rx1, ry1, rx2, ry2 = (int(round(value)) for value in region)
        rx1 = max(0, min(frame_width, rx1))
        ry1 = max(0, min(frame_height, ry1))
        rx2 = max(0, min(frame_width, rx2))
        ry2 = max(0, min(frame_height, ry2))

        if rx2 <= rx1 or ry2 <= ry1:
            return []

        cropped = np.ascontiguousarray(prepared[ry1:ry2, rx1:rx2])
        crop_height, crop_width = cropped.shape[:2]
        if max_dimension is not None:
            scale = max(
                1.0,
                min(
                    scale,
                    max_dimension / max(crop_width, crop_height),
                ),
            )
        target_width = max(1, int(round(crop_width * scale)))
        target_height = max(1, int(round(crop_height * scale)))

        if target_width == crop_width and target_height == crop_height:
            working = cropped
        else:
            working = cv2.resize(
                cropped,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        scale_x = target_width / crop_width
        scale_y = target_height / crop_height
        if score_threshold is None:
            local_detections = self.detect_faces(working)
        else:
            previous_threshold = self._detector.getScoreThreshold()
            self._detector.setScoreThreshold(float(score_threshold))
            try:
                local_detections = self.detect_faces(working)
            finally:
                self._detector.setScoreThreshold(previous_threshold)
        mapped: list[FaceDetection] = []

        for detection in local_detections:
            raw = np.asarray(detection.raw, dtype=np.float32).copy()
            if raw.size < 15:
                continue

            raw[0] = raw[0] / scale_x + rx1
            raw[1] = raw[1] / scale_y + ry1
            raw[2] = raw[2] / scale_x
            raw[3] = raw[3] / scale_y

            for x_index in (4, 6, 8, 10, 12):
                raw[x_index] = raw[x_index] / scale_x + rx1
                raw[x_index + 1] = (
                    raw[x_index + 1] / scale_y + ry1
                )

            x, y, width, height = (
                float(value) for value in raw[:4]
            )
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(frame_width, int(round(x + width)))
            y2 = min(frame_height, int(round(y + height)))

            if x2 <= x1 or y2 <= y1:
                continue

            mapped.append(
                FaceDetection(
                    box=(x1, y1, x2, y2),
                    score=detection.score,
                    raw=np.ascontiguousarray(raw),
                )
            )

        return sorted(mapped, key=lambda item: item.score, reverse=True)

    @staticmethod
    def validate_face_size(
        detection: FaceDetection,
        minimum_size: int,
    ) -> None:
        if detection.width < minimum_size or detection.height < minimum_size:
            raise FaceTooSmallError(
                f"Face is {detection.width}x{detection.height}px; "
                f"minimum is {minimum_size}x{minimum_size}px."
            )

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size not in (128, 512):
            raise EmbeddingGenerationError(
                f"Expected 128 or 512 embedding dimension, got {vector.size} values."
            )
        if not np.all(np.isfinite(vector)):
            raise EmbeddingGenerationError("Embedding contains invalid values.")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise EmbeddingGenerationError("Embedding has zero length.")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def embedding_from_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        try:
            feature = self._recognizer.feature(self._prepare(aligned_face))
        except cv2.error as exc:
            raise EmbeddingGenerationError(
                "SFace failed to generate an embedding."
            ) from exc
        return self._normalize(feature)

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> FaceEmbeddingResult:
        prepared = self._prepare(image)
        try:
            aligned = self._recognizer.alignCrop(prepared, detection.raw)
        except cv2.error as exc:
            raise EmbeddingGenerationError("SFace alignment failed.") from exc
        if aligned is None or aligned.size == 0:
            raise EmbeddingGenerationError("SFace returned an empty crop.")
        embedding = self.embedding_from_aligned(aligned)
        return FaceEmbeddingResult(detection, aligned, embedding)

    def extract_single_face(
        self,
        image: np.ndarray,
        *,
        minimum_face_size: int,
    ) -> FaceEmbeddingResult:
        detections = self.detect_faces(image)
        if not detections:
            raise NoFaceDetectedError("No face detected.")
        if len(detections) != 1:
            raise MultipleFacesDetectedError(
                f"Expected one face, detected {len(detections)}."
            )
        self.validate_face_size(detections[0], minimum_face_size)
        return self.extract_embedding(image, detections[0])