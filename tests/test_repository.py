import numpy as np

from src.repository import FaceRepository


def test_embedding_blob_round_trip() -> None:
    vector = np.arange(1, 129, dtype=np.float32)
    blob = FaceRepository.serialize_embedding(vector)
    restored = FaceRepository.deserialize_embedding(blob)
    assert restored.shape == (128,)
    assert restored.dtype == np.float32
    assert np.isclose(np.linalg.norm(restored), 1.0)
