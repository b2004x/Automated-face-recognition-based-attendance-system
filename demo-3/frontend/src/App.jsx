// frontend/src/App.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/yolo";

const todayISO = () => new Date().toISOString().slice(0, 10);
const b64Img = (b64) => (b64 ? `data:image/jpeg;base64,${b64}` : "");

function DataTable({ columns, rows, emptyText = "Chưa có dữ liệu." }) {
  if (!rows || rows.length === 0) {
    return <div className="notice">{emptyText}</div>;
  }
  return (
    <div className="tableWrap">
      <table className="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.title}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx}>
              {columns.map((c) => (
                <td key={c.key}>{r[c.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MsgBox({ type, children }) {
  if (!children) return null;
  const cls =
    type === "ok"
      ? "msg ok"
      : type === "warn"
      ? "msg warn"
      : type === "err"
      ? "msg err"
      : "msg info";
  return <div className={cls}>{children}</div>;
}

export default function App() {
  /** ===== tabs ===== */
  const [tab, setTab] = useState("enroll"); // enroll | att

  /** ===== sidebar controls (Streamlit like) ===== */
  const [yoloConf, setYoloConf] = useState(0.4);
  const [yoloIou, setYoloIou] = useState(0.45);
  const [resizeW, setResizeW] = useState(640);
  const [yoloSkip, setYoloSkip] = useState(2);
  const [maxDet, setMaxDet] = useState(10);

  const [expandScale, setExpandScale] = useState(1.6);
  const [minSide, setMinSide] = useState(200);

  /** ===== camera/overlay ===== */
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const capCanvasRef = useRef(null);

  /** ===== ws ===== */
  const wsRef = useRef(null);
  const lastBBoxRef = useRef(null);
  const lastConfRef = useRef(null);
  const [wsStatus, setWsStatus] = useState("—");

  /** ===== enroll state ===== */
  const [enName, setEnName] = useState("");
  const [enId, setEnId] = useState("");
  const [enClass, setEnClass] = useState("");
  const [enMsg, setEnMsg] = useState({ type: "", text: "" });
  const [enCropB64, setEnCropB64] = useState("");
  const [enrolledRows, setEnrolledRows] = useState([]);

  /** ===== attendance state ===== */
  const today = useMemo(() => todayISO(), []);
  const [threshold, setThreshold] = useState(0.45);
  const [attMsg, setAttMsg] = useState({ type: "", text: "" });
  const [attCropB64, setAttCropB64] = useState("");
  const [attJson, setAttJson] = useState(null);

  const [logsToday, setLogsToday] = useState([]);
  const [reportDay, setReportDay] = useState(today);
  const [summaryRows, setSummaryRows] = useState([]);
  const [counts, setCounts] = useState({ present: 0, absent: 0 });

  /** ===== draw overlay ===== */
  function drawOverlay(bbox, conf) {
    const v = videoRef.current;
    const c = overlayRef.current;
    if (!v || !c) return;

    const vw = v.videoWidth || 640;
    const vh = v.videoHeight || 480;

    c.width = vw;
    c.height = vh;

    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, vw, vh);

    if (!bbox) return;
    const [x1, y1, x2, y2] = bbox;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#22c55e";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const text = `YOLO ${(conf ?? 0).toFixed(2)}`;
    ctx.font = "16px ui-sans-serif, system-ui";
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(
      x1,
      Math.max(0, y1 - 24),
      ctx.measureText(text).width + 14,
      22
    );

    ctx.fillStyle = "white";
    ctx.fillText(text, x1 + 7, Math.max(16, y1 - 8));
  }

  /** ===== camera ===== */
  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
  }
  function stopCamera() {
    const stream = videoRef.current?.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
  }

  /** ===== capture JPEG (resized like Streamlit resize_w) ===== */
  async function captureJpegBlob(quality = 0.88) {
    const v = videoRef.current;
    const canvas = capCanvasRef.current;
    if (!v || !canvas) return null;

    const vw = v.videoWidth || 640;
    const vh = v.videoHeight || 480;

    const targetW = Number(resizeW || vw);
    const scale = targetW / vw;
    const targetH = Math.max(1, Math.round(vh * scale));

    canvas.width = targetW;
    canvas.height = targetH;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(v, 0, 0, targetW, targetH);

    return new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", quality)
    );
  }

  /** ===== WS connect (supports config) ===== */
  function connectWS() {
    try {
      wsRef.current?.close();
    } catch {}
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setWsStatus("✅ connected");
    ws.onclose = () => setWsStatus("⚠️ closed");
    ws.onerror = () => setWsStatus("❌ error");

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        lastBBoxRef.current = data.bbox || null;
        lastConfRef.current = data.conf ?? null;
        drawOverlay(data.bbox || null, data.conf ?? null);
      } catch {}
    };
  }

  function sendWSConfig() {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== 1) return;
    ws.send(
      JSON.stringify({
        type: "config",
        conf: yoloConf,
        iou: yoloIou,
        max_det: maxDet,
        skip_n: yoloSkip,
      })
    );
  }

  async function wsSendLoop() {
    while (true) {
      await new Promise((r) => setTimeout(r, 150));
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) continue;

      const blob = await captureJpegBlob(0.85);
      if (!blob) continue;

      const buf = await blob.arrayBuffer();
      ws.send(buf);
    }
  }

  /** ===== backend fetches ===== */
  async function fetchEnrolled(limit = 30) {
    const res = await fetch(`${API}/enrolled/recent?limit=${limit}`);
    if (!res.ok) return;
    const data = await res.json();
    setEnrolledRows(data.rows || []);
  }

  async function fetchLogs(day, limit = 50) {
    const res = await fetch(`${API}/attendance/logs?day=${day}&limit=${limit}`);
    if (!res.ok) return;
    const data = await res.json();
    setLogsToday(data.rows || []);
  }

  async function fetchSummary(day) {
    const res = await fetch(`${API}/attendance/summary?day=${day}`);
    if (!res.ok) return;
    const data = await res.json();
    setCounts({ present: data.present ?? 0, absent: data.absent ?? 0 });
    setSummaryRows(
      (data.rows || []).map((r) => ({
        student_id: r[0],
        name: r[1],
        class_name: r[2],
        status: r[3],
        score: r[4],
        timestamp: r[5],
      }))
    );
  }

  /** ===== reset database ===== */
  async function resetDatabase() {
    const ok = confirm(
      "Bạn chắc chắn muốn RESET database?\nHành động này sẽ xoá toàn bộ sinh viên + log điểm danh."
    );
    if (!ok) return;

    const res = await fetch(`${API}/db/reset`, { method: "POST" });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      alert(`Reset failed: ${data.error || "unknown"}`);
      return;
    }

    // refresh UI
    setEnCropB64("");
    setAttCropB64("");
    setAttJson(null);
    setEnMsg({ type: "info", text: "ℹ️ Đã reset database." });
    setAttMsg({ type: "info", text: "ℹ️ Đã reset database." });

    fetchEnrolled(30);
    fetchLogs(today, 50);
    fetchSummary(reportDay);
  }

  async function resetAttendanceLogs(dayStr) {
  const ok = confirm(
    `Bạn chắc chắn muốn XOÁ LOG ĐIỂM DANH của ngày ${dayStr}?\n(Sinh viên đã đăng ký sẽ giữ nguyên.)`
  );
  if (!ok) return;

  const res = await fetch(`${API}/attendance/reset?day=${encodeURIComponent(dayStr)}`, {
    method: "POST",
  });
  const data = await res.json().catch(() => ({}));

  if (!res.ok || !data.ok) {
    alert(`Reset logs failed: ${data.error || "unknown"}`);
    return;
  }

  // refresh UI đúng phần điểm danh
  setAttCropB64("");
  setAttJson(null);
  setAttMsg({ type: "info", text: `ℹ️ Đã xoá log điểm danh ngày ${dayStr}.` });

  // reload log + summary + (nếu dayStr=today thì log hôm nay cũng rỗng)
  fetchLogs(today, 50);
  fetchSummary(reportDay);
}

  /** ===== init ===== */
  useEffect(() => {
    (async () => {
      await startCamera();
      connectWS();
      wsSendLoop();

      fetchEnrolled(30);
      fetchLogs(today, 50);
      fetchSummary(reportDay);
    })();

    return () => {
      try {
        wsRef.current?.close();
      } catch {}
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** whenever sliders change => send config to WS */
  useEffect(() => {
    sendWSConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [yoloConf, yoloIou, yoloSkip, maxDet]);

  useEffect(() => {
    fetchSummary(reportDay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reportDay]);

  /** ===== actions ===== */
  async function onEnrollSave() {
    setEnMsg({ type: "", text: "" });
    setEnCropB64("");

    if (!enName.trim() || !enId.trim() || !enClass.trim()) {
      setEnMsg({ type: "err", text: "❌ Vui lòng nhập đủ: Họ tên, MSSV, Lớp." });
      return;
    }
    const bbox = lastBBoxRef.current;
    if (!bbox) {
      setEnMsg({
        type: "err",
        text: "❌ YOLO chưa detect mặt. Đưa mặt vào khung rồi bấm Lưu.",
      });
      return;
    }

    const blob = await captureJpegBlob(0.9);
    if (!blob) {
      setEnMsg({ type: "err", text: "❌ Chưa capture được frame." });
      return;
    }

    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");
    fd.append("student_id", enId.trim());
    fd.append("name", enName.trim());
    fd.append("class_name", enClass.trim());
    fd.append("bbox_x1", bbox[0]);
    fd.append("bbox_y1", bbox[1]);
    fd.append("bbox_x2", bbox[2]);
    fd.append("bbox_y2", bbox[3]);
    fd.append("yolo_conf", lastConfRef.current ?? 0);
    fd.append("expand_scale", expandScale);
    fd.append("min_side", minSide);

    const res = await fetch(`${API}/enroll`, { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      setEnMsg({ type: "err", text: `❌ ${data.error || "Enroll failed"}` });
      if (data.crop_b64) setEnCropB64(data.crop_b64);
      return;
    }

    setEnMsg({ type: "ok", text: "✅ Đã lưu (UPSERT theo MSSV)." });
    if (data.crop_b64) setEnCropB64(data.crop_b64);

    fetchEnrolled(30);
  }

  async function onAttend() {
    setAttMsg({ type: "", text: "" });
    setAttCropB64("");
    setAttJson(null);

    const bbox = lastBBoxRef.current;
    if (!bbox) {
      setAttMsg({
        type: "err",
        text: "❌ YOLO chưa detect mặt. Đưa mặt vào khung rồi bấm Điểm danh.",
      });
      return;
    }

    const blob = await captureJpegBlob(0.9);
    if (!blob) {
      setAttMsg({ type: "err", text: "❌ Chưa capture được frame." });
      return;
    }

    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");
    fd.append("bbox_x1", bbox[0]);
    fd.append("bbox_y1", bbox[1]);
    fd.append("bbox_x2", bbox[2]);
    fd.append("bbox_y2", bbox[3]);
    fd.append("threshold", threshold);
    fd.append("expand_scale", expandScale);
    fd.append("min_side", minSide);

    const res = await fetch(`${API}/attend`, { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      setAttMsg({ type: "err", text: `❌ ${data.error || "Attend failed"}` });
      if (data.crop_b64) setAttCropB64(data.crop_b64);
      return;
    }

    if (data.crop_b64) setAttCropB64(data.crop_b64);
    setAttJson(data);

    if (data.matched === false) {
      setAttMsg({
        type: "warn",
        text: `⚠️ Không khớp ai (best_score=${Number(
          data.best_score ?? 0
        ).toFixed(2)} < threshold=${Number(
          data.threshold ?? threshold
        ).toFixed(2)})`,
      });
    } else {
      setAttMsg({
        type: data.logged ? "ok" : "info",
        text: data.logged
          ? `✅ Điểm danh thành công: ${data.student_id} | ${data.name} | ${data.class_name} | score=${Number(
              data.score
            ).toFixed(2)}`
          : `ℹ️ ${data.student_id} đã được điểm danh hôm nay rồi | score=${Number(
              data.score
            ).toFixed(2)}`,
      });
    }

    fetchLogs(today, 50);
    fetchSummary(reportDay);
  }

  function downloadCsv() {
    window.open(`${API}/attendance/csv?day=${reportDay}`, "_blank");
  }

  /** ===== tables ===== */
  const enrolledTable = enrolledRows.map((r) => ({
    student_id: r[0],
    name: r[1],
    class_name: r[2],
    updated_at: r[3],
  }));

  const logsTable = logsToday.map((r) => ({
    student_id: r[0],
    name: r[1],
    class_name: r[2],
    score: r[3],
    ts: r[4],
  }));

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="headerTitle">
            <div className="emoji">👤</div>
            <div>
              <h1>Face System</h1>
              <p>Đăng ký & Điểm danh (YOLO → crop → InsightFace → cosine)</p>
            </div>
          </div>

          <div className="pill">
            WS: <b>{wsStatus}</b>
          </div>
        </header>

        <div className="layout">
          {/* ===== Sidebar ===== */}
          <aside className="sidebar">
            <h3>YOLO realtime</h3>

            <div className="control">
              <div className="labelRow">
                <span>YOLO conf</span>
                <span className="value">{yoloConf.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={yoloConf}
                onChange={(e) => setYoloConf(parseFloat(e.target.value))}
              />
            </div>

            <div className="control">
              <div className="labelRow">
                <span>YOLO IoU</span>
                <span className="value">{yoloIou.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.95"
                step="0.05"
                value={yoloIou}
                onChange={(e) => setYoloIou(parseFloat(e.target.value))}
              />
            </div>

            <div className="control">
              <div className="labelRow">
                <span>Resize width</span>
              </div>
              <select
                value={resizeW}
                onChange={(e) => setResizeW(parseInt(e.target.value, 10))}
              >
                {[1280, 960, 800, 640, 480].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>

            <div className="control">
              <div className="labelRow">
                <span>YOLO infer mỗi N frame</span>
              </div>
              <select
                value={yoloSkip}
                onChange={(e) => setYoloSkip(parseInt(e.target.value, 10))}
              >
                {[1, 2, 3, 4, 5].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>

            <div className="control">
              <div className="labelRow">
                <span>Max detections</span>
              </div>
              <input
                type="number"
                min="1"
                max="50"
                value={maxDet}
                onChange={(e) =>
                  setMaxDet(parseInt(e.target.value || "10", 10))
                }
              />
            </div>

            <div className="divider" />

            <h3>Crop/InsightFace (khi bấm nút)</h3>

            <div className="control">
              <div className="labelRow">
                <span>Nới bbox (scale)</span>
                <span className="value">{expandScale.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="2.2"
                step="0.05"
                value={expandScale}
                onChange={(e) => setExpandScale(parseFloat(e.target.value))}
              />
            </div>

            <div className="control">
              <div className="labelRow">
                <span>Upscale nếu min side &lt;</span>
                <span className="value">{minSide}px</span>
              </div>
              <input
                type="range"
                min="120"
                max="420"
                step="10"
                value={minSide}
                onChange={(e) => setMinSide(parseInt(e.target.value, 10))}
              />
            </div>

            <div className="hint">
              Slider YOLO sẽ gửi config realtime qua WebSocket (cần backend patch).
            </div>
          </aside>

          {/* ===== Main ===== */}
          <main className="main">
            <div className="tabs">
              <button
                className={`tab ${tab === "enroll" ? "active" : ""}`}
                onClick={() => setTab("enroll")}
              >
                📝 Đăng ký (UPSERT theo MSSV)
              </button>
              <button
                className={`tab ${tab === "att" ? "active" : ""}`}
                onClick={() => setTab("att")}
              >
                ✅ Điểm danh + Tải danh sách
              </button>
            </div>

            <div className="mainGrid">
              {/* LEFT: Camera */}
              <section className="card">
                <div className="cardTitle">Camera (YOLO bbox)</div>

                <div className="cameraWrap">
                  <video ref={videoRef} className="video" playsInline />
                  <canvas ref={overlayRef} className="overlay" />
                </div>

                <canvas ref={capCanvasRef} className="hidden" />

                <div className="meta">
                  ResizeW={resizeW} | skip={yoloSkip} | conf={yoloConf.toFixed(2)}{" "}
                  | iou={yoloIou.toFixed(2)} | max_det={maxDet}
                </div>
              </section>

              {/* RIGHT: Panel */}
              {tab === "enroll" ? (
                <div className="rightCol">
                  <section className="card">
                    <div className="cardTitle">Thông tin sinh viên</div>

                    <div className="form">
                      <input
                        placeholder="Họ và tên"
                        value={enName}
                        onChange={(e) => setEnName(e.target.value)}
                      />
                      <input
                        placeholder="MSSV"
                        value={enId}
                        onChange={(e) => setEnId(e.target.value)}
                      />
                      <input
                        placeholder="Lớp"
                        value={enClass}
                        onChange={(e) => setEnClass(e.target.value)}
                      />

                      <button className="btn primary" onClick={onEnrollSave}>
                        💾 Lưu / Cập nhật
                      </button>

                      {/* ✅ reset db in enroll tab */}
                      <button className="btn danger" onClick={resetDatabase}>
                        🗑️ Reset database
                      </button>

                      <MsgBox type={enMsg.type}>{enMsg.text}</MsgBox>
                    </div>

                    <div className="divider" />

                    <div className="cardTitle">Crop preview (khi bấm Lưu)</div>
                    {enCropB64 ? (
                      <img
                        src={b64Img(enCropB64)}
                        alt="crop"
                        className="previewImg"
                      />
                    ) : (
                      <div className="notice">Chưa có</div>
                    )}
                  </section>

                  <section className="card">
                    <div className="rowBetween">
                      <div className="cardTitle">
                        Danh sách đã đăng ký (30 bản ghi gần nhất)
                      </div>
                      <button className="btn" onClick={() => fetchEnrolled(30)}>
                        🔄 Reload
                      </button>
                    </div>

                    <DataTable
                      columns={[
                        { key: "student_id", title: "student_id" },
                        { key: "name", title: "name" },
                        { key: "class_name", title: "class_name" },
                        { key: "updated_at", title: "updated_at" },
                      ]}
                      rows={enrolledTable}
                      emptyText="Chưa có sinh viên nào."
                    />
                  </section>
                </div>
              ) : (
                <div className="rightCol">
                  <section className="card">
                    <div className="cardTitle">Thiết lập nhận diện</div>

                    <div className="control">
                      <div className="labelRow">
                        <span>Ngưỡng cosine</span>
                        <span className="value">{threshold.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.2"
                        max="0.8"
                        step="0.01"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      />
                    </div>

                    <button className="btn primary" onClick={onAttend}>
                      ✅ Điểm danh
                    </button>

                    {/* ✅ reset db in attendance tab */}
                    <button
                    className="btn danger"
                    onClick={() => resetAttendanceLogs(reportDay)}
                  >
                    🧹 Xoá log CSV (ngày đang chọn)
                  </button>

                    <MsgBox type={attMsg.type}>{attMsg.text}</MsgBox>

                    <div className="divider" />

                    <div className="cardTitle">Ảnh crop trước khi điểm danh</div>
                    {attCropB64 ? (
                      <img
                        src={b64Img(attCropB64)}
                        alt="crop"
                        className="previewImg"
                      />
                    ) : (
                      <div className="notice">Chưa có</div>
                    )}

                    <div className="divider" />

                    <div className="cardTitle">Kết quả</div>
                    {attJson ? (
                      <pre className="json">{JSON.stringify(attJson, null, 2)}</pre>
                    ) : (
                      <div className="notice">Chưa có</div>
                    )}
                  </section>

                  <section className="card">
                    <div className="rowBetween">
                      <div className="cardTitle">Log điểm danh hôm nay</div>
                      <button className="btn" onClick={() => fetchLogs(today, 50)}>
                        🔄 Reload
                      </button>
                    </div>

                    <DataTable
                      columns={[
                        { key: "student_id", title: "student_id" },
                        { key: "name", title: "name" },
                        { key: "class_name", title: "class_name" },
                        { key: "score", title: "score" },
                        { key: "ts", title: "ts" },
                      ]}
                      rows={logsTable}
                      emptyText="Chưa có log hôm nay."
                    />

                    <div className="divider" />

                    <div className="cardTitle">
                      📥 Tải xuống danh sách (đã/ chưa điểm danh)
                    </div>

                    <div className="report">
                      <div>
                        <div className="labelRow">
                          <span>Chọn ngày xuất danh sách</span>
                        </div>
                        <input
                          type="date"
                          value={reportDay}
                          onChange={(e) => setReportDay(e.target.value)}
                        />
                      </div>

                      <div className="row">
                        <button className="btn" onClick={() => fetchSummary(reportDay)}>
                          🔄 Reload danh sách
                        </button>
                        <button className="btn" onClick={downloadCsv}>
                          ⬇️ Tải CSV
                        </button>
                      </div>

                      <div className="notice">
                        ✅ Đã điểm danh: <b>{counts.present}</b> | ❌ Chưa điểm danh:{" "}
                        <b>{counts.absent}</b>
                      </div>

                      <DataTable
                        columns={[
                          { key: "student_id", title: "student_id" },
                          { key: "name", title: "name" },
                          { key: "class_name", title: "class_name" },
                          { key: "status", title: "status" },
                          { key: "score", title: "score" },
                          { key: "timestamp", title: "timestamp" },
                        ]}
                        rows={summaryRows}
                      />
                    </div>
                  </section>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
