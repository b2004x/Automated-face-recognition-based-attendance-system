# app.py
# 2 TAB:
# 1) 📝 Đăng ký: YOLO realtime vẽ bbox, CHỈ khi bấm LƯU mới crop (+nới bbox + upscale) -> InsightFace embedding -> UPSERT theo MSSV
# 2) ✅ Điểm danh: YOLO realtime vẽ bbox, CHỈ khi bấm "✅ Điểm danh" mới crop -> hiện ảnh crop preview -> embedding -> cosine -> ghi log
#    + Tải xuống danh sách (đã/ chưa điểm danh) theo ngày (CSV)
#
# Cài:
#   pip install streamlit streamlit-webrtc av opencv-python ultralytics insightface onnxruntime numpy
# Chạy:
#   streamlit run app.py

import streamlit as st
import cv2
import av
import numpy as np
import sqlite3
from datetime import datetime, date
from threading import Lock
import csv, io

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo
from deepface import DeepFace

# =========================
# CONFIG
# =========================
DB_PATH = "faces.db"
YOLO_MODEL_PATH = r"..\model\Face_detection\yolov12n-face.pt"

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")  # UTC+7

# =========================
# DB
# =========================
def db_connect():
    return sqlite3.connect(DB_PATH)

def now_vn():
    return datetime.now(VN_TZ)

def init_db():
    conn = db_connect()
    cur = conn.cursor()

    # main table
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

    # auto-migrate columns if old DB
    cur.execute("PRAGMA table_info(face_embeddings)")
    cols = [r[1] for r in cur.fetchall()]
    if "det_score" not in cols:
        cur.execute("ALTER TABLE face_embeddings ADD COLUMN det_score REAL")
    if "updated_at" not in cols:
        cur.execute("ALTER TABLE face_embeddings ADD COLUMN updated_at TEXT")
    conn.commit()

    # DEDUPE before creating UNIQUE index (prevents IntegrityError if you had duplicates)
    cur.execute("""
        DELETE FROM face_embeddings
        WHERE id NOT IN (
            SELECT MAX(id) FROM face_embeddings GROUP BY student_id
        )
    """)
    conn.commit()

    # unique index for UPSERT
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_face_embeddings_student_id
        ON face_embeddings(student_id)
    """)

    # attendance logs
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
    # 1 log / student / day
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
    return ok


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



def fetch_enrolled_list(limit=30):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT student_id, name, class_name, updated_at
        FROM face_embeddings
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows



def get_attendance_summary(day_str: str):
    """
    Return rows:
      student_id, name, class_name, status, score, timestamp
    status: present/absent
    """
    conn = db_connect()
    cur = conn.cursor()

    # logs for day
    cur.execute("""
        SELECT student_id, score, ts
        FROM attendance_logs
        WHERE day = ?
    """, (day_str,))
    log_map = {sid: (float(score), ts) for (sid, score, ts) in cur.fetchall()}

    # all enrolled students
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


def summary_to_csv_bytes(rows):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["student_id", "name", "class_name", "status", "score", "timestamp"])
    w.writerows(rows)
    return out.getvalue().encode("utf-8-sig")  # excel-friendly

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
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.fuse()
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    return yolo, fa

yolo_model, fa = load_models()
init_db()


# =========================
# VISION UTILS
# =========================
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


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-12)


# =========================
# VIDEO PROCESSOR (YOLO overlay only)
# =========================
class YoloOverlayProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = Lock()
        self.latest_frame = None
        self.latest_bbox = None
        self.latest_conf = None

        self.conf_thres = 0.4
        self.iou_thres = 0.45
        self.max_det = 10
        self.resize_width = 640
        self.skip_n = 2
        self.frame_id = 0
        self._last_best = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h0, w0 = img.shape[:2]

        if self.resize_width and w0 != self.resize_width:
            s = self.resize_width / w0
            img = cv2.resize(img, (self.resize_width, int(h0 * s)), interpolation=cv2.INTER_LINEAR)

        self.frame_id += 1
        if self.frame_id % self.skip_n == 0:
            results = yolo_model.predict(
                source=img,
                conf=float(self.conf_thres),
                iou=float(self.iou_thres),
                verbose=False,
                max_det=int(self.max_det),
            )
            self._last_best = pick_best_yolo_box(results)

        best = self._last_best
        if best is not None:
            x1, y1, x2, y2, conf = best
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"YOLO {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self.lock:
                self.latest_frame = img.copy()
                self.latest_bbox = (x1, y1, x2, y2)
                self.latest_conf = float(conf)
        else:
            with self.lock:
                self.latest_frame = img.copy()
                self.latest_bbox = None
                self.latest_conf = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# UI
# =========================
st.set_page_config(page_title="Enroll + Attendance (click-to-recognize + download)", layout="wide")
st.title("👤 Face System: Đăng ký & Điểm danh (YOLO → crop → InsightFace → cosine)")

# sidebar controls
with st.sidebar:
    st.header("YOLO realtime")
    yolo_conf = st.slider("YOLO conf", 0.05, 0.95, 0.40, 0.05)
    yolo_iou = st.slider("YOLO IoU", 0.10, 0.95, 0.45, 0.05)
    resize_w = st.selectbox("Resize width", [1280, 960, 800, 640, 480], index=3)
    yolo_skip = st.selectbox("YOLO infer mỗi N frame", [1, 2, 3, 4, 5], index=1)
    max_det = st.slider("Max detections", 1, 50, 10, 1)

    st.divider()
    st.header("Crop/InsightFace (khi bấm nút)")
    expand_scale = st.slider("Nới bbox (scale)", 1.0, 2.2, 1.6, 0.05)
    min_side = st.slider("Upscale crop nếu min side < (px)", 120, 420, 200, 10)

tab_enroll, tab_att = st.tabs(["📝 Đăng ký (UPSERT theo MSSV)", "✅ Điểm danh + Tải danh sách"])


# =========================
# TAB 1: ENROLL
# =========================
with tab_enroll:
    c1, c2 = st.columns([2, 1], gap="large")

    with c2:
        st.subheader("Thông tin sinh viên")
        name = st.text_input("Họ và tên", key="en_name")
        student_id = st.text_input("MSSV", key="en_id")
        class_name = st.text_input("Lớp", key="en_class")
        save_btn = st.button("💾 Lưu / Cập nhật", type="primary", width='stretch', key="en_save")
        status = st.empty()

        st.divider()
        st.subheader("Crop preview (khi bấm Lưu)")
        preview = st.empty()

        st.divider()
        st.subheader("Danh sách đã đăng ký (30 bản ghi gần nhất)")

        # ---- Reload button ----
        if st.button("🔄 Reload danh sách đã đăng ký", width='stretch', key="reload_enrolled"):
            st.session_state["enrolled_rows"] = fetch_enrolled_list(30)

        # cache lần đầu
        if "enrolled_rows" not in st.session_state:
            st.session_state["enrolled_rows"] = fetch_enrolled_list(30)

        rows = st.session_state["enrolled_rows"]
        if rows:
            st.dataframe(rows, width='stretch', hide_index=True)
        else:
            st.info("Chưa có sinh viên nào.")

    with c1:
        st.subheader("Camera (YOLO bbox)")
        ctx_en = webrtc_streamer(
            key="cam-enroll",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YoloOverlayProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    # apply settings to processor
    if ctx_en.video_processor:
        vp = ctx_en.video_processor
        vp.conf_thres = float(yolo_conf)
        vp.iou_thres = float(yolo_iou)
        vp.resize_width = int(resize_w)
        vp.skip_n = int(yolo_skip)
        vp.max_det = int(max_det)

    # handle save (only now do crop + insightface)
    if save_btn:
        if not name.strip() or not student_id.strip() or not class_name.strip():
            status.error("❌ Vui lòng nhập đủ: Họ tên, MSSV, Lớp.")
        elif not ctx_en.video_processor:
            status.error("❌ Camera chưa sẵn sàng.")
        else:
            with ctx_en.video_processor.lock:
                frame_bgr = None if ctx_en.video_processor.latest_frame is None else ctx_en.video_processor.latest_frame.copy()
                bbox = ctx_en.video_processor.latest_bbox
                yconf = ctx_en.video_processor.latest_conf
        
            if frame_bgr is None:
                status.error("❌ Chưa có frame.")
            elif bbox is None:
                status.error("❌ YOLO chưa detect mặt. Đưa mặt vào khung rồi bấm Lưu.")
            else:
                h, w = frame_bgr.shape[:2]
                bb = clamp_bbox(*bbox, w=w, h=h)
                if bb is None:
                    status.error("❌ BBox không hợp lệ.")
                else:
                    bb2 = expand_bbox(bb, w, h, scale=float(expand_scale)) or bb
                    x1, y1, x2, y2 = bb2
                    crop = frame_bgr[y1:y2, x1:x2].copy()
                    crop = upscale_if_small(crop, min_side=int(min_side))
                    
                    preview.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                  caption=f"Crop: {crop.shape} | scale={expand_scale}",
                                  width='stretch')
                    is_real, spoof_score = run_antispoof(crop)
                    if is_real is False:
                        status.error("❌ Phát hiện sử dụng ảnh giả mạo (spoofing detected).")
                    elif is_real is None:
                        status.error("⚠️ Không xác định được anti-spoof. Đưa mặt rõ hơn.")
                    else:
                        faces = fa.get(crop)
                        if not faces:
                            status.error("❌ InsightFace không detect mặt trong crop. Tăng scale (1.8~2.0) hoặc min_side (240).")
                        else:
                            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                            f0 = faces[0]
                            emb = getattr(f0, "embedding", None)
                            det_score = float(getattr(f0, "det_score", 0.0))

                            if emb is None or len(emb) == 0:
                                status.error("❌ Không lấy được embedding.")
                            else:
                                emb = l2_normalize(np.asarray(emb, dtype=np.float32))
                                upsert_student(
                                    student_id=student_id,
                                    name=name,
                                    class_name=class_name,
                                    yolo_conf=yconf,
                                    det_score=det_score,
                                    bbox=bb2,
                                    embedding_vec=emb,
                                    crop_bgr=crop
                                )
                                status.success("✅ Đã lưu (UPSERT theo MSSV).")
                                st.session_state["enrolled_rows"] = fetch_enrolled_list(30)
                                st.session_state["emb_cache"] = load_embeddings()


# =========================
# TAB 2: ATTENDANCE (CLICK-TO-RECOGNIZE + DOWNLOAD)
# =========================
with tab_att:
    c1, c2 = st.columns([2, 1], gap="large")

    with c2:
        st.subheader("Thiết lập nhận diện")
        threshold = st.slider("Ngưỡng cosine", 0.20, 0.80, 0.45, 0.01, key="att_th")

        if "emb_cache" not in st.session_state:
            st.session_state["emb_cache"] = load_embeddings()

        reload_btn = st.button("🔄 Reload embeddings", width='stretch', key="att_reload")
        if reload_btn:
            st.session_state["emb_cache"] = load_embeddings()

        ids, names, classes, mat = st.session_state["emb_cache"]
        st.write(f"Đã nạp: **{len(ids)}** sinh viên")
        if mat is None:
            st.warning("Chưa có embeddings trong DB. Hãy qua tab Đăng ký để lưu trước.")

        st.divider()
        st.subheader("Bấm để điểm danh")
        att_btn = st.button("✅ Điểm danh", type="primary", width='stretch', key="att_btn")

        st.subheader("Ảnh crop trước khi điểm danh")
        crop_preview = st.empty()

        st.subheader("Kết quả")
        result_box = st.empty()

        st.divider()
        st.subheader("Log điểm danh hôm nay")
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT student_id, name, class_name, score, ts
            FROM attendance_logs
            WHERE day = ?
            ORDER BY ts DESC
            LIMIT 50
        """, (date.today().isoformat(),))
        logs = cur.fetchall()
        conn.close()
        if logs:
            st.dataframe(logs, width='stretch', hide_index=True)
        else:
            st.info("Chưa có log hôm nay.")

        # ===== Download summary (present + absent) =====
        st.divider()
        st.subheader("📥 Tải xuống danh sách (đã/ chưa điểm danh)")
        report_day = st.date_input("Chọn ngày xuất danh sách", value=now_vn().date(), key="report_day").isoformat()

        # ---- Reload summary button ----
        if st.button("🔄 Reload danh sách đã/chưa điểm danh", width='stretch', key="reload_summary"):
            st.session_state["summary_rows"] = get_attendance_summary(report_day)

        # cache lần đầu hoặc khi đổi ngày
        if ("summary_rows" not in st.session_state) or (st.session_state.get("summary_day") != report_day):
            st.session_state["summary_rows"] = get_attendance_summary(report_day)
            st.session_state["summary_day"] = report_day

        summary_rows = st.session_state["summary_rows"]
        present_count = sum(1 for r in summary_rows if r[3] == "present")
        absent_count = sum(1 for r in summary_rows if r[3] == "absent")
        st.write(f"✅ Đã điểm danh: **{present_count}** | ❌ Chưa điểm danh: **{absent_count}**")

        st.dataframe(summary_rows, width='stretch', hide_index=True)

        csv_bytes = summary_to_csv_bytes(summary_rows)
        st.download_button(
            "⬇️ Tải CSV (present/absent)",
            data=csv_bytes,
            file_name=f"attendance_{report_day}.csv",
            mime="text/csv",
            width='stretch'
        )


    with c1:
        st.subheader("Camera (YOLO bbox)")
        ctx_att = webrtc_streamer(
            key="cam-attendance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YoloOverlayProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    # apply YOLO settings
    if ctx_att.video_processor:
        vp = ctx_att.video_processor
        vp.conf_thres = float(yolo_conf)
        vp.iou_thres = float(yolo_iou)
        vp.resize_width = int(resize_w)
        vp.skip_n = int(yolo_skip)
        vp.max_det = int(max_det)

    # When click "Điểm danh": capture latest frame & bbox, crop preview, embed, cosine, log
    if att_btn:
        if mat is None or len(ids) == 0:
            result_box.error("❌ Chưa có dữ liệu đăng ký. Hãy đăng ký trước.")
        elif not ctx_att.video_processor:
            result_box.error("❌ Camera chưa sẵn sàng.")
        else:
            with ctx_att.video_processor.lock:
                frame_bgr = None if ctx_att.video_processor.latest_frame is None else ctx_att.video_processor.latest_frame.copy()
                bbox = ctx_att.video_processor.latest_bbox

            if frame_bgr is None:
                result_box.error("❌ Chưa có frame.")
            elif bbox is None:
                result_box.error("❌ YOLO chưa detect mặt. Đưa mặt vào khung rồi bấm Điểm danh.")
            else:
                h, w = frame_bgr.shape[:2]
                bb = clamp_bbox(*bbox, w=w, h=h)
                if bb is None:
                    result_box.error("❌ BBox không hợp lệ.")
                else:
                    bb2 = expand_bbox(bb, w, h, scale=float(expand_scale)) or bb
                    x1, y1, x2, y2 = bb2
                    crop = frame_bgr[y1:y2, x1:x2].copy()
                    crop = upscale_if_small(crop, min_side=int(min_side))

                    # show crop BEFORE recognition
                    crop_preview.image(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        caption=f"Crop: {crop.shape} | scale={expand_scale}",
                        width='stretch'
                    )
                    is_real, spoof_score = run_antispoof(crop)
                    if is_real is False:
                        result_box.error("❌ Phát hiện spoofing – vui lòng dùng mặt thật.")
                    elif is_real is None:
                        result_box.warning("⚠️ Không xác định được anti-spoof.")
                    else:
                        faces = fa.get(crop)
                        if not faces:
                            result_box.error("❌ InsightFace không detect mặt trong crop. Tăng scale/min_side hoặc nhìn thẳng camera.")
                        else:
                            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                            f0 = faces[0]
                            emb = getattr(f0, "embedding", None)
                            if emb is None or len(emb) == 0:
                                result_box.error("❌ Không lấy được embedding.")
                            else:
                                q = l2_normalize(np.asarray(emb, dtype=np.float32))
                                scores = (mat @ q).astype(np.float32)  # cosine via dot
                                idx = int(np.argmax(scores))
                                best_score = float(scores[idx])

                                if best_score < float(threshold):
                                    result_box.warning(f"⚠️ Không khớp ai (best_score={best_score:.2f} < threshold={threshold:.2f})")
                                else:
                                    sid = ids[idx]
                                    nm = names[idx]
                                    cl = classes[idx]
                                    logged = insert_attendance_if_new_day(sid, nm, cl, best_score)
                                    current_day = report_day  # hoặc now_vn().date().isoformat() nếu bạn dùng TZ
                                    st.session_state["summary_rows"] = get_attendance_summary(current_day)
                                    st.session_state["summary_day"] = current_day

                                    if logged:
                                        result_box.success(f"✅ Điểm danh thành công: {sid} | {nm} | {cl} | score={best_score:.2f}")
                                    else:
                                        result_box.info(f"ℹ️ {sid} đã được điểm danh hôm nay rồi | score={best_score:.2f}")
