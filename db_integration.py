import os
import re
import uuid
import time
import logging
import numpy as np
import psycopg2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Errors here were previously silently swallowed after DATABASE_RETRY_COUNT
# retries, which is how column-name mismatches like this went unnoticed.
# Now every failure also lands in logs/db_integration.log for investigation.
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_logger = logging.getLogger("db_integration")
if not _logger.handlers:
    _handler = logging.FileHandler(_LOG_DIR / "db_integration.log")
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Configuration Variables
# RECOGNITION_BUFFER_SECONDS: The amount of time in seconds to wait after a track is first seen 
#                             before committing its identity to the database. This allows the recognition
#                             system to stabilize. Default: 3.0. Recommended range: 2.0 - 5.0.
RECOGNITION_BUFFER_SECONDS = float(os.getenv("RECOGNITION_BUFFER_SECONDS", "3.0"))

# VISITOR_SIMILARITY_THRESHOLD: The minimum cosine similarity required to match a new unknown face
#                               to an existing visitor in the database. Default: 0.60. Recommended range: 0.50 - 0.75.
VISITOR_SIMILARITY_THRESHOLD = float(os.getenv("VISITOR_SIMILARITY_THRESHOLD", "0.60"))

# DATABASE_RETRY_COUNT: Number of times to retry database operations if the connection drops. Default: 3.
DATABASE_RETRY_COUNT = int(os.getenv("DATABASE_RETRY_COUNT", "3"))

DATABASE_URL = os.getenv("DATABASE_URL")

def _encode_track_id(cam_name: str, track_id: int) -> int:
    """
    track_id is a bigint[] column, not text[] -- it can't hold strings like
    "CAM1_5". To still keep track ids distinguishable per camera (so the
    same numeric track_id on two different cameras isn't treated as the
    same sighting), encode camera + track_id into one integer:
        cam_number * 1_000_000 + track_id
    e.g. CAM1 track 5 -> 1000005, CAM3 track 5 -> 3000005.
    Falls back to just track_id if no digit is found in cam_name.
    """
    match = re.search(r"(\d+)", cam_name)
    cam_number = int(match.group(1)) if match else 0
    return cam_number * 1_000_000 + int(track_id)


def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in .env")
    return psycopg2.connect(DATABASE_URL)

def add_track_id_to_ai_team(name: str, cam_name: str, track_id: int):
    """
    Appends the camera-encoded track_id (see _encode_track_id) to the
    ai_team track_id array if it does not already exist.
    """
    track_val = _encode_track_id(cam_name, track_id)
    for _ in range(DATABASE_RETRY_COUNT):
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Append track_val only if it's not already in the array.
                    # Case-insensitive: FaceMatcher's identity name comes from
                    # folder names on disk (e.g. lowercase "sarisht"), which
                    # never reliably matches the capitalized name stored here
                    # (e.g. "Sarisht") on an exact-case comparison.
                    cur.execute("""
                        UPDATE public.ai_team
                        SET track_id = array_append(track_id, %s::bigint)
                        WHERE LOWER(name) = LOWER(%s) AND NOT (%s = ANY(track_id));
                    """, (track_val, name, track_val))
                    
                    # If the array was null, array_append on null returns an array with 1 element
                    # Wait, if track_id is NULL, NOT (x = ANY(NULL)) evaluates to NULL (falsy in WHERE).
                    # So we need to handle NULL arrays explicitly.
                    cur.execute("""
                        UPDATE public.ai_team
                        SET track_id = ARRAY[%s]::bigint[]
                        WHERE LOWER(name) = LOWER(%s) AND track_id IS NULL;
                    """, (track_val, name))
            break
        except Exception as e:
            print(f"DB Error ai_team update: {e}")
            _logger.error("add_track_id_to_ai_team failed | name=%s cam=%s track=%s | %s", name, cam_name, track_id, e)
            time.sleep(0.5)

def handle_unknown_visitor(cam_name: str, track_id: int, embedding: np.ndarray) -> str:
    """
    Finds the most similar existing visitor or creates a new one, 
    and appends the track_id.
    """
    if embedding is None:
        # No embedding means we can't dedupe or persist this person at all.
        # Caller (camera.py) should not invoke this until a best_embedding
        # exists for the track. Log it since a random UUID here is never
        # written to visitors and will silently vanish otherwise.
        _logger.warning(
            "handle_unknown_visitor called with no embedding | cam=%s track=%s",
            cam_name, track_id,
        )
        return str(uuid.uuid4())
        
    track_val = _encode_track_id(cam_name, track_id)
    best_match_id = None
    best_sim = -1.0
    
    for _ in range(DATABASE_RETRY_COUNT):
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT face_id, embeddings FROM public.visitors;")
                    visitors = cur.fetchall()
                    
                    for v_id, v_emb in visitors:
                        if v_emb is None:
                            continue
                        v_emb_np = np.array(v_emb, dtype=np.float32)
                        sim = float(np.dot(embedding, v_emb_np))
                        if sim > best_sim:
                            best_sim = sim
                            best_match_id = v_id
                            
                    if best_sim >= VISITOR_SIMILARITY_THRESHOLD and best_match_id is not None:
                        # Update existing
                        cur.execute("""
                            UPDATE public.visitors
                            SET track_id = array_append(track_id, %s::bigint)
                            WHERE face_id = %s AND NOT (%s = ANY(track_id));
                        """, (track_val, best_match_id, track_val))
                        
                        cur.execute("""
                            UPDATE public.visitors
                            SET track_id = ARRAY[%s]::bigint[]
                            WHERE face_id = %s AND track_id IS NULL;
                        """, (track_val, best_match_id))
                        return str(best_match_id)
                    else:
                        # Insert new
                        new_id = str(uuid.uuid4())
                        cur.execute("""
                            INSERT INTO public.visitors (face_id, name, phone, age, designation, created_at, track_id, embeddings)
                            VALUES (%s, %s, %s, %s, %s, %s, ARRAY[%s]::bigint[], %s);
                        """, (
                            new_id, 
                            "Unknown", 
                            None, 
                            None, 
                            "Visitor", 
                            datetime.now(), 
                            track_val, 
                            embedding.tolist()
                        ))
                        return new_id
            break
        except Exception as e:
            print(f"DB Error visitors update: {e}")
            _logger.error("handle_unknown_visitor failed | cam=%s track=%s | %s", cam_name, track_id, e)
            time.sleep(0.5)
            
    return str(uuid.uuid4())

def insert_visitor_log(person_a_name_or_id: str, person_b_name_or_id: str, start_time: float, end_time: float, comments: str = ""):
    """
    Resolves names/UUIDs to ai_team_face_id / visitor_face_id and logs the
    conversation, using the visitor_logs table exactly as it exists today
    (log_id, visitor_face_id, ai_team_face_id, start_time, end_time, comments,
    created_at) -- no schema changes.

    AI_TEAM <-> VISITOR : one row, ai_team_face_id + visitor_face_id both set.

    VISITOR <-> VISITOR : the table only has one visitor_face_id column, so
    there's no single row that can hold both people. Instead this inserts
    TWO rows, one per visitor (ai_team_face_id left NULL in both), each with the
    other visitor's face_id appended into comments so the pairing is still
    fully investigable -- e.g. querying visitor_logs by either person's
    visitor_face_id finds their side of the conversation.

    AI_TEAM <-> AI_TEAM : not logged (caller already filters this out).
    """
    def resolve_id(name_or_id):
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check AI team. Case-insensitive: see note in
                # add_track_id_to_ai_team about folder-name vs DB-name casing.
                cur.execute("SELECT face_id FROM public.ai_team WHERE LOWER(name) = LOWER(%s);", (name_or_id,))
                res = cur.fetchone()
                if res:
                    return True, res[0]
                
                # Check visitors
                try:
                    uuid_val = str(uuid.UUID(name_or_id))
                    cur.execute("SELECT face_id FROM public.visitors WHERE face_id = %s;", (uuid_val,))
                    res = cur.fetchone()
                    if res:
                        return False, res[0]
                except ValueError:
                    pass
        return None, None

    def _insert_row(ai_team_face_id, visitor_face_id, row_comments):
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO public.visitor_logs
                        (ai_team_face_id, visitor_face_id, start_time, end_time, created_at, comments)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (ai_team_face_id, visitor_face_id, start_dt, end_dt, datetime.now(), row_comments))

    for _ in range(DATABASE_RETRY_COUNT):
        try:
            a_is_ai, a_id = resolve_id(person_a_name_or_id)
            b_is_ai, b_id = resolve_id(person_b_name_or_id)

            if a_id is None and b_id is None:
                return  # Can't log if nobody found

            if a_id is None or b_id is None:
                # One side genuinely didn't resolve to any known ai_team or
                # visitors row. Treating the unresolved side as a second
                # visitor here is exactly what caused a NOT NULL violation
                # on ai_team_face_id previously (a real AI team member whose
                # name lookup failed got silently miscategorized as an
                # unresolved visitor). Skip and log instead of guessing.
                _logger.warning(
                    "SKIPPED visitor_log | could not resolve one side | a=%s (resolved=%s) b=%s (resolved=%s)",
                    person_a_name_or_id, a_id is not None,
                    person_b_name_or_id, b_id is not None,
                )
                return

            if a_is_ai and b_is_ai:
                # AI <-> AI: not part of this feature, skip silently.
                return

            if not a_is_ai and not b_is_ai:
                # VISITOR <-> VISITOR: two rows, cross-referenced in comments.
                _insert_row(None, a_id, f"{comments} | with_visitor={b_id}")
                _insert_row(None, b_id, f"{comments} | with_visitor={a_id}")
                _logger.info(
                    "SYNCED visitor_log (VISITOR<->VISITOR, 2 rows) | a=%s b=%s",
                    a_id, b_id,
                )
            else:
                # AI <-> VISITOR: single row, both columns used as designed.
                ai_team_face_id = a_id if a_is_ai else b_id
                visitor_face_id = b_id if a_is_ai else a_id
                _insert_row(ai_team_face_id, visitor_face_id, comments)
                _logger.info(
                    "SYNCED visitor_log (AI<->VISITOR) | ai_team_face_id=%s visitor_face_id=%s",
                    ai_team_face_id, visitor_face_id,
                )
            break
        except Exception as e:
            print(f"DB Error visitor_logs insert: {e}")
            _logger.error(
                "insert_visitor_log failed | a=%s b=%s | %s",
                person_a_name_or_id, person_b_name_or_id, e,
            )
            time.sleep(0.5)