"""Hardware Acceleration Module: Auto-configures PyTorch CUDA GPU devices,
half-precision FP16 execution, and safe OpenCV DNN backends."""

from __future__ import annotations

import cv2
import numpy as np

# PyTorch & CUDA Check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE_STR = "cuda:0" if CUDA_AVAILABLE else "cpu"
    USE_FP16 = CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE_STR = "cpu"
    USE_FP16 = False


def get_hardware_info() -> dict[str, str | bool]:
    """Returns details about available GPU hardware acceleration."""
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "device": DEVICE_STR,
        "use_fp16": USE_FP16,
    }
    if CUDA_AVAILABLE:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = "NVIDIA CUDA GPU"
    else:
        info["gpu_name"] = "CPU Execution"
    return info


def configure_opencv_dnn(net) -> bool:
    """
    Safely configures an OpenCV DNN net or FaceDetectorYN/FaceRecognizerSF instance
    for CUDA/GPU acceleration with automatic CPU fallback if unsupported.
    """
    if net is None:
        return False

    if CUDA_AVAILABLE:
        try:
            if hasattr(net, "setPreferableBackend") and hasattr(net, "setPreferableTarget"):
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                return True
        except Exception:
            pass  # Fall through to default OpenCV CPU backend below

    try:
        if hasattr(net, "setPreferableBackend") and hasattr(net, "setPreferableTarget"):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass

    return False


def get_yolo_device_kwargs() -> dict[str, str | bool]:
    """Returns keyword arguments for Ultralytics YOLO model calls."""
    if CUDA_AVAILABLE:
        return {"device": 0, "half": True}
    return {"device": "cpu", "half": False}
