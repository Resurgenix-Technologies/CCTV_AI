import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# CONFIG
# -----------------------

TEST_IMAGE = "image.png"

PERSON_FILE = "data/persons.json"
REGISTRY_DIR = "registry"

# -----------------------
# LOAD FACE MODEL
# -----------------------

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------
# LOAD PERSONS
# -----------------------

with open(PERSON_FILE, "r") as f:
    persons = json.load(f)

# -----------------------
# LOAD TEST IMAGE
# -----------------------

img = cv2.imread(TEST_IMAGE)

faces = app.get(img)

if len(faces) == 0:
    raise Exception("No face detected")

face = max(
    faces,
    key=lambda x:
    (x.bbox[2]-x.bbox[0]) *
    (x.bbox[3]-x.bbox[1])
)

query_embedding = face.embedding.reshape(1, -1)

# -----------------------
# MATCH
# -----------------------

best_score = -1
best_id = None

for global_id in persons:

    emb_path = os.path.join(
        REGISTRY_DIR,
        f"{global_id}.npy"
    )

    db_embedding = np.load(emb_path).reshape(1, -1)

    score = cosine_similarity(
        query_embedding,
        db_embedding
    )[0][0]

    print(
        f"{global_id} : {score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_id = global_id




# -----------------------
# RESULT
# -----------------------

print("\n==========")
print("BEST MATCH")
print("==========")

print("Global ID:", best_id)
print("Name:", persons[best_id]["name"])
print("Similarity:", round(best_score, 4))