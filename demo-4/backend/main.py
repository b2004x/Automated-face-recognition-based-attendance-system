import io
import csv
import base64
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pathlib import Path

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from deepface import DeepFace

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov12n-face.pt"
YOLO_MODEL_PATH = MODEL_DIR / MODEL_NAME

DB_PATH = "faces.db"
# YOLO_MODEL_PATH = r"model\yolov12n-face.pt"  # sửa đúng đường dẫn của bạn
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# =========================
# APP + CORS
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DB
# =========================
def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def now_vn():
    return datetime.now(VN_TZ)

def init_db():
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,

            yolo_conf REAL,
            det_score REAL,

            bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,

            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,

            crop_jpg BLOB,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("PRAGMA table_info(face_embeddings)")
    cols = [r[1] for r in cur.fetchall()]
    if "det_score" not in cols:
        cur.execute("ALTER TABLE face_embeddings ADD COLUMN det_score REAL")
    if "updated_at" not in cols:
        cur.execute("ALTER TABLE face_embeddings ADD COLUMN updated_at TEXT")
    conn.commit()

    # dedupe by student_id
    cur.execute("""
        DELETE FROM face_embeddings
        WHERE id NOT IN (SELECT MAX(id) FROM face_embeddings GROUP BY student_id)
    """)
    conn.commit()

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_face_embeddings_student_id
        ON face_embeddings(student_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            score REAL NOT NULL,
            ts TEXT NOT NULL,
            day TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_unique_day
        ON attendance_logs(student_id, day)
    """)
    conn.commit()
    conn.close()

def to_jpg_bytes(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None

def img_to_b64_jpg(img_bgr) -> str:
    b = to_jpg_bytes(img_bgr)
    return base64.b64encode(b).decode("utf-8") if b else ""

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-12)

def upsert_student(student_id, name, class_name, yolo_conf, det_score, bbox, embedding_vec, crop_bgr):
    conn = db_connect()
    cur = conn.cursor()

    emb = np.asarray(embedding_vec, dtype=np.float32).flatten()
    emb_bytes = emb.tobytes()
    dim = int(emb.shape[0])

    crop_jpg = to_jpg_bytes(crop_bgr)
    x1, y1, x2, y2 = bbox
    now = now_vn().isoformat(timespec="seconds")

    cur.execute("""
        INSERT INTO face_embeddings (
            student_id, name, class_name,
            yolo_conf, det_score,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            embedding, embedding_dim, crop_jpg,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            name=excluded.name,
            class_name=excluded.class_name,
            yolo_conf=excluded.yolo_conf,
            det_score=excluded.det_score,
            bbox_x1=excluded.bbox_x1,
            bbox_y1=excluded.bbox_y1,
            bbox_x2=excluded.bbox_x2,
            bbox_y2=excluded.bbox_y2,
            embedding=excluded.embedding,
            embedding_dim=excluded.embedding_dim,
            crop_jpg=excluded.crop_jpg,
            updated_at=excluded.updated_at
    """, (
        student_id.strip(), name.strip(), class_name.strip(),
        float(yolo_conf) if yolo_conf is not None else None,
        float(det_score) if det_score is not None else None,
        int(x1), int(y1), int(x2), int(y2),
        emb_bytes, dim, crop_jpg,
        now
    ))

    conn.commit()
    conn.close()

def insert_attendance_if_new_day(student_id, name, class_name, score):
    conn = db_connect()
    cur = conn.cursor()
    ts = now_vn().isoformat(timespec="seconds")
    day_str = now_vn().date().isoformat()
    try:
        cur.execute("""
            INSERT INTO attendance_logs(student_id, name, class_name, score, ts, day)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, name, class_name, float(score), ts, day_str))
        conn.commit()
        ok = True
    except sqlite3.IntegrityError:
        ok = False
    conn.close()
    return ok, day_str, ts

def load_embeddings():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT student_id, name, class_name, embedding_dim, embedding
        FROM face_embeddings
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return [], [], [], None

    ids, names, classes, vecs = [], [], [], []
    dim_ref = None
    for sid, nm, cls, dim, blob in rows:
        d = int(dim)
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != d:
            continue
        if dim_ref is None:
            dim_ref = d
        if d != dim_ref:
            continue
        ids.append(sid)
        names.append(nm)
        classes.append(cls)
        vecs.append(v)

    if not vecs:
        return [], [], [], None

    mat = np.stack(vecs, axis=0).astype(np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return ids, names, classes, mat

def get_attendance_summary(day_str: str):
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT student_id, score, ts
        FROM attendance_logs
        WHERE day = ?
    """, (day_str,))
    log_map = {sid: (float(score), ts) for (sid, score, ts) in cur.fetchall()}

    cur.execute("""
        SELECT student_id, name, class_name
        FROM face_embeddings
        ORDER BY class_name ASC, student_id ASC
    """)
    students = cur.fetchall()
    conn.close()

    rows = []
    for sid, nm, cl in students:
        if sid in log_map:
            sc, ts = log_map[sid]
            rows.append((sid, nm, cl, "present", sc, ts))
        else:
            rows.append((sid, nm, cl, "absent", "", ""))
    return rows

def rows_to_csv_bytes(rows):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["student_id", "name", "class_name", "status", "score", "timestamp"])
    w.writerows(rows)
    return out.getvalue().encode("utf-8-sig")

# =========================
# VISION UTILS
# =========================
def decode_jpg_bytes(b: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def expand_bbox(bbox, w, h, scale=1.6):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    nbw = bw * scale
    nbh = bh * scale
    nx1 = int(cx - nbw / 2)
    ny1 = int(cy - nbh / 2)
    nx2 = int(cx + nbw / 2)
    ny2 = int(cy + nbh / 2)
    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(0, min(w - 1, nx2))
    ny2 = max(0, min(h - 1, ny2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)

def upscale_if_small(img, min_side=200):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    ms = min(h, w)
    if ms >= min_side:
        return img
    s = min_side / ms
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)

def pick_best_yolo_box(results):
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    best = None
    for b in r.boxes:
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cand = (x1, y1, x2, y2, conf)
        if best is None or conf > best[4]:
            best = cand
    return best

def run_antispoof(crop_bgr, min_size=80):
    """
    Return: (is_real: bool | None, score: float | None)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None

    h, w = crop_bgr.shape[:2]
    if h < min_size or w < min_size:
        return None, None

    # DeepFace thích RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    try:
        res = DeepFace.extract_faces(
            img_path=crop_rgb,
            anti_spoofing=True,
            enforce_detection=False
        )
        if not res:
            return None, None

        score = float(res[0].get("antispoof_score", 0.0))
        is_real = bool(res[0].get("is_real", False))
        return is_real, score

    except Exception as e:
        print("Anti-spoof error:", e)
        return None, None

# =========================
# MODELS
# =========================
yolo_model = YOLO(YOLO_MODEL_PATH)
try:
    yolo_model.fuse()
except Exception:
    pass

fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))  # bạn thấy log "set det-size"

init_db()

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.websocket("/ws/yolo")
async def ws_yolo(websocket: WebSocket):
    """
    Client gửi JPEG bytes liên tục.
    Server trả JSON: {"bbox":[x1,y1,x2,y2], "conf":0.78, "w":..., "h":...} hoặc {"bbox":null}
    """
    await websocket.accept()

    # default config (có thể mở rộng để client gửi config sau)
    conf_thres = 0.4
    iou_thres = 0.45
    max_det = 10
    skip_n = 2

    frame_id = 0
    last_best = None

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_id += 1

            img = decode_jpg_bytes(data)
            if img is None:
                await websocket.send_json({"bbox": None})
                continue

            h, w = img.shape[:2]

            if frame_id % skip_n == 0:
                results = yolo_model.predict(
                    source=img,
                    conf=float(conf_thres),
                    iou=float(iou_thres),
                    verbose=False,
                    max_det=int(max_det),
                )
                last_best = pick_best_yolo_box(results)

            if last_best is None:
                await websocket.send_json({"bbox": None, "w": w, "h": h})
            else:
                x1, y1, x2, y2, c = last_best
                await websocket.send_json({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(c),
                    "w": w, "h": h
                })

    except WebSocketDisconnect:
        return

@app.post("/enroll")
async def enroll(
    image: UploadFile = File(...),
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(...),
    bbox_x1: int = Form(...),
    bbox_y1: int = Form(...),
    bbox_x2: int = Form(...),
    bbox_y2: int = Form(...),
    yolo_conf: Optional[float] = Form(None),
    expand_scale: float = Form(1.6),
    min_side: int = Form(200),
):
    b = await image.read()
    frame = decode_jpg_bytes(b)
    if frame is None:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)

    h, w = frame.shape[:2]
    bb = clamp_bbox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, w, h)
    if bb is None:
        return JSONResponse({"ok": False, "error": "invalid bbox"}, status_code=400)

    bb2 = expand_bbox(bb, w, h, scale=float(expand_scale)) or bb
    x1, y1, x2, y2 = bb2

    crop_raw = frame[y1:y2, x1:x2].copy()
    crop_for_model = upscale_if_small(crop_raw.copy(), min_side=int(min_side))

    is_real, spoof_score = run_antispoof(crop_for_model)
    if is_real is False:
        return JSONResponse({"ok": False,"error": "❌ Phát hiện spoofing – vui lòng dùng mặt thật."})
    elif is_real is None:
        return JSONResponse({"ok": False,"error": "⚠️ Không xác định được anti-spoof."})
    
    faces = fa.get(crop_for_model)
    if not faces:
        return {"ok": False, "error": "insightface no face in crop", "crop_b64": img_to_b64_jpg(crop_raw)}

    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    f0 = faces[0]
    emb = getattr(f0, "embedding", None)
    det_score = float(getattr(f0, "det_score", 0.0))

    if emb is None or len(emb) == 0:
        return {"ok": False, "error": "no embedding", "crop_b64": img_to_b64_jpg(crop_raw)}

    emb = l2_normalize(np.asarray(emb, dtype=np.float32))

    # lưu crop_for_model (hoặc muốn lưu raw thì đổi crop_raw)
    upsert_student(
        student_id=student_id,
        name=name,
        class_name=class_name,
        yolo_conf=yolo_conf,
        det_score=det_score,
        bbox=bb2,
        embedding_vec=emb,
        crop_bgr=crop_for_model,
    )

    return {
        "ok": True,
        "student_id": student_id,
        "name": name,
        "class_name": class_name,
        "det_score": det_score,
        "crop_b64": img_to_b64_jpg(crop_raw),  # preview không phóng to ảo
        "updated_at": now_vn().isoformat(timespec="seconds"),
    }

@app.post("/attend")
async def attend(
    image: UploadFile = File(...),
    bbox_x1: int = Form(...),
    bbox_y1: int = Form(...),
    bbox_x2: int = Form(...),
    bbox_y2: int = Form(...),
    threshold: float = Form(0.45),
    expand_scale: float = Form(1.6),
    min_side: int = Form(200),
):
    ids, names, classes, mat = load_embeddings()
    if mat is None or len(ids) == 0:
        return JSONResponse({"ok": False, "error": "no enrolled embeddings"}, status_code=400)

    b = await image.read()
    frame = decode_jpg_bytes(b)
    if frame is None:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)

    h, w = frame.shape[:2]
    bb = clamp_bbox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, w, h)
    if bb is None:
        return JSONResponse({"ok": False, "error": "invalid bbox"}, status_code=400)

    bb2 = expand_bbox(bb, w, h, scale=float(expand_scale)) or bb
    x1, y1, x2, y2 = bb2

    crop_raw = frame[y1:y2, x1:x2].copy()
    crop_for_model = upscale_if_small(crop_raw.copy(), min_side=int(min_side))
    
    is_real, spoof_score = run_antispoof(crop_for_model)
    if is_real is False:
        return JSONResponse({"ok": False,"error": "❌ Phát hiện sử dụng ảnh giả mạo (spoofing detected)."})
    elif is_real is None:
        return JSONResponse({"ok": False,"error": "⚠️ Không xác định được anti-spoof. Đưa mặt rõ hơn."})
    
    faces = fa.get(crop_for_model)
    if not faces:
        return {"ok": False, "error": "insightface no face in crop", "crop_b64": img_to_b64_jpg(crop_raw)}

    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    f0 = faces[0]
    emb = getattr(f0, "embedding", None)
    if emb is None or len(emb) == 0:
        return {"ok": False, "error": "no embedding", "crop_b64": img_to_b64_jpg(crop_raw)}

    q = l2_normalize(np.asarray(emb, dtype=np.float32))
    scores = (mat @ q).astype(np.float32)
    idx = int(np.argmax(scores))
    best_score = float(scores[idx])

    if best_score < float(threshold):
        return {
            "ok": True,
            "matched": False,
            "best_score": best_score,
            "threshold": float(threshold),
            "crop_b64": img_to_b64_jpg(crop_raw),
        }

    sid = ids[idx]
    nm = names[idx]
    cl = classes[idx]
    logged, day_str, ts = insert_attendance_if_new_day(sid, nm, cl, best_score)

    return {
        "ok": True,
        "matched": True,
        "student_id": sid,
        "name": nm,
        "class_name": cl,
        "score": best_score,
        "logged": logged,
        "day": day_str,
        "ts": ts,
        "crop_b64": img_to_b64_jpg(crop_raw),
    }

@app.get("/attendance/summary")
def attendance_summary(day: str):
    rows = get_attendance_summary(day)
    present = sum(1 for r in rows if r[3] == "present")
    absent = sum(1 for r in rows if r[3] == "absent")
    return {"day": day, "present": present, "absent": absent, "rows": rows}

@app.get("/attendance/csv")
def attendance_csv(day: str):
    rows = get_attendance_summary(day)
    data = rows_to_csv_bytes(rows)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attendance_{day}.csv"'},
    )
    
@app.get("/enrolled/recent")
def enrolled_recent(limit: int = 30):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT student_id, name, class_name, updated_at
        FROM face_embeddings
        ORDER BY updated_at DESC
        LIMIT ?
    """, (int(limit),))
    rows = cur.fetchall()
    conn.close()
    return {"rows": rows}

@app.get("/attendance/logs")
def attendance_logs(day: str, limit: int = 50):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT student_id, name, class_name, score, ts
        FROM attendance_logs
        WHERE day = ?
        ORDER BY ts DESC
        LIMIT ?
    """, (day, int(limit)))
    rows = cur.fetchall()
    conn.close()
    return {"rows": rows}

@app.post("/db/reset")
def db_reset():
    try:
        conn = db_connect()
        cur = conn.cursor()

        # xóa dữ liệu
        cur.execute("DELETE FROM attendance_logs")
        cur.execute("DELETE FROM face_embeddings")
        conn.commit()
        conn.close()

        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

from fastapi import Query
from fastapi.responses import JSONResponse

@app.post("/attendance/reset")
def reset_attendance_logs(day: str = Query(..., description="YYYY-MM-DD")):
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM attendance_logs WHERE day = ?", (day,))
        conn.commit()
        conn.close()
        return {"ok": True, "day": day}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

