"""SCRFD/YuNet face detection, SFace alignment and embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from src.low_light import lime_enhance


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


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """One detected face and its normalized embedding."""

    detection: FaceDetection
    aligned_face: np.ndarray
    embedding: np.ndarray


class FaceEngine:
    """Reusable face detector with the existing SFace recognizer."""

    def __init__(
        self,
        detector_model: Path,
        recognizer_model: Path,
        *,
        detection_score_threshold: float = 0.85,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        detector_backend: str = "yunet",
        low_light_enhancement: str = "none",
    ) -> None:
        self.detector_model = detector_model.resolve()
        self.recognizer_model = recognizer_model.resolve()
        self.detector_backend = detector_backend.strip().lower()
        self.detection_score_threshold = float(detection_score_threshold)
        self.low_light_enhancement = low_light_enhancement.strip().lower()
        if self.low_light_enhancement not in {"none", "lime"}:
            raise ValueError("low_light_enhancement must be 'none' or 'lime'.")
        if self.detector_backend not in {"yunet", "scrfd"}:
            raise ValueError("detector_backend must be 'yunet' or 'scrfd'.")

        self._validate_model(self.detector_model, "YuNet")
        self._validate_model(self.recognizer_model, "SFace")

        try:
            if self.detector_backend == "scrfd":
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],
                    providers=["CPUExecutionProvider"],
                )
                # Larger SCRFD inference canvas improves recall for distant
                # CCTV faces. The per-track ROI path still bounds the work.
                app.prepare(ctx_id=-1, det_size=(960, 960))
                self._scrfd = app.det_model
                self._detector = None
            else:
                self._scrfd = None
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
            encoded = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error) as exc:
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
        detector_image = (
            lime_enhance(prepared)
            if self.low_light_enhancement == "lime"
            else prepared
        )
        height, width = detector_image.shape[:2]

        if self.detector_backend == "scrfd":
            try:
                boxes, landmarks = self._scrfd.detect(
                    detector_image,
                    max_num=0,
                    metric="default",
                )
            except Exception as exc:
                raise FaceEngineError("SCRFD failed to process the frame.") from exc
            detections: list[FaceDetection] = []
            if boxes is None or landmarks is None:
                return detections
            for box, points in zip(boxes, landmarks, strict=True):
                score = float(box[4])
                if score < self.detection_score_threshold:
                    continue
                x1 = max(0, int(round(float(box[0]))))
                y1 = max(0, int(round(float(box[1]))))
                x2 = min(width, int(round(float(box[2]))))
                y2 = min(height, int(round(float(box[3]))))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw = np.zeros(15, dtype=np.float32)
                raw[:4] = (x1, y1, x2 - x1, y2 - y1)
                raw[4:14] = np.asarray(points, dtype=np.float32).reshape(-1)[:10]
                raw[14] = score
                detections.append(FaceDetection((x1, y1, x2, y2), score, raw))
            return sorted(detections, key=lambda item: item.score, reverse=True)

        assert self._detector is not None
        self._detector.setInputSize((width, height))

        try:
            _, faces = self._detector.detect(detector_image)
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

        ``upscale`` is used only to help YuNet locate small faces. When
        ``max_dimension`` is set, enlargement is reduced so the longest
        working side stays within that budget whenever the native crop is
        smaller; native pixels are never downscaled. Bounding boxes and all
        five landmarks are mapped back to the original frame so
        :meth:`extract_embedding` still aligns from the real source pixels.
        Enlarged pixels therefore never count as additional identity detail.
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
        elif getattr(self, "detector_backend", "yunet") == "scrfd":
            previous_threshold = self.detection_score_threshold
            self.detection_score_threshold = float(score_threshold)
            try:
                local_detections = self.detect_faces(working)
            finally:
                self.detection_score_threshold = previous_threshold
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
        if vector.size != EMBEDDING_DIMENSION:
            raise EmbeddingGenerationError(
                f"Expected {EMBEDDING_DIMENSION} values, got {vector.size}."
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

    @staticmethod
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), angle, 1.0
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _jpeg_round_trip(
        image: np.ndarray,
        quality: int = 50,
    ) -> np.ndarray:
        """Apply deterministic JPEG artifacts without changing dimensions."""

        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100.")
        prepared = FaceEngine._prepare(image)
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            prepared,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not encoded_ok:
            raise EmbeddingGenerationError(
                "Could not generate the CCTV JPEG augmentation."
            )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise EmbeddingGenerationError(
                "Could not decode the CCTV JPEG augmentation."
            )
        return np.ascontiguousarray(decoded)

    def embedding_variants(
        self,
        aligned_face: np.ndarray,
        count: int,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Generate mild deterministic enrollment augmentations.

        These are fallback representations from the same source image,
        not independent photographs. The compression variant models the
        block artifacts present in distant CCTV samples while keeping the
        original aligned geometry unchanged.
        """

        candidates: list[tuple[str, np.ndarray]] = [
            ("original", aligned_face),
            (
                "cctv_jpeg_50",
                self._jpeg_round_trip(aligned_face, quality=50),
            ),
            ("horizontal_flip", cv2.flip(aligned_face, 1)),
            (
                "slightly_brighter",
                cv2.convertScaleAbs(aligned_face, alpha=1.04, beta=8),
            ),
            (
                "slightly_darker",
                cv2.convertScaleAbs(aligned_face, alpha=0.96, beta=-8),
            ),
            ("rotate_plus_3", self._rotate(aligned_face, 3.0)),
            ("rotate_minus_3", self._rotate(aligned_face, -3.0)),
        ]

        selected = candidates[: max(1, min(count, len(candidates)))]
        return [
            (name, self.embedding_from_aligned(variant))
            for name, variant in selected
        ]
