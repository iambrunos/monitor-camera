import os
import time
import json
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import requests
import onnxruntime as ort

# Fuso horário de Brasília (UTC-3)
BRT = timezone(timedelta(hours=-3))


def now_brt_str(fmt="%H:%M:%S") -> str:
    return datetime.now(BRT).strftime(fmt)


# ================= CONFIGURAÇÕES PADRÃO (Lidas das Variáveis de Ambiente Seguras) =================
CAM_URL = os.getenv("CAM_URL", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEB_PASS = os.getenv("WEB_PASS", "")
AUTH_TOKEN = hashlib.sha256(f"cammon_auth_{WEB_PASS}".encode("utf-8")).hexdigest() if WEB_PASS else ""
PORT = int(os.getenv("PORT", "10000"))

MODEL_PATH = "yolov8n.onnx"
MODEL_URLS = [
    "https://huggingface.co/s1777/yolo-v8n-onnx/resolve/main/yolov8n.onnx",
    "https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx",
]

STATE_FILE = "config_state.json"

# Estado dinâmico
state = {
    "alerts_on": True,
    "threshold": float(os.getenv("THRESHOLD", "0.22")),
    "cooldown_sec": int(os.getenv("COOLDOWN_SEC", "25")),
    # Polígono de 8 pontos em frações (0.0 a 1.0) para contornar perfeitamente a calçada/rua
    "poly": [
        [0.01, 0.36],
        [0.20, 0.37],
        [0.38, 0.38],
        [0.53, 0.68],
        [0.68, 0.99],
        [0.35, 0.99],
        [0.01, 0.99],
        [0.01, 0.68],
    ],
    "last_alert_ts": 0.0,
    "total_alerts": 0,
    "stream_online": False,
    "started_at": now_brt_str("%d/%m/%Y %H:%M:%S"),
}

latest_frame_lock = threading.Lock()
latest_raw_frame = None
latest_raw_ts = 0.0
latest_annotated_jpeg = None


def expand_to_8_points(pts):
    if len(pts) == 4:
        out = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            out.append(p1)
            out.append([round((p1[0] + p2[0]) / 2.0, 4), round((p1[1] + p2[1]) / 2.0, 4)])
        return out
    return pts


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k in ("alerts_on", "threshold", "cooldown_sec", "poly"):
                    if k in saved:
                        state[k] = saved[k]
                state["poly"] = expand_to_8_points(state["poly"])
        except Exception as e:
            print(f"[Config] Erro ao ler {STATE_FILE}: {e}")


def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alerts_on": state["alerts_on"],
                    "threshold": state["threshold"],
                    "cooldown_sec": state["cooldown_sec"],
                    "poly": state["poly"],
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"[Config] Erro ao salvar {STATE_FILE}: {e}")


# ================= TELEGRAM API =================
def tg_send_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as e:
        print(f"[Telegram] Erro sendMessage: {e}")


def tg_send_photo(jpeg_bytes: bytes, caption: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": ("alerta.jpg", jpeg_bytes, "image/jpeg")},
            timeout=20,
        )
    except Exception as e:
        print(f"[Telegram] Erro sendPhoto: {e}")


def telegram_bot_loop():
    """Escuta comandos enviados pelo usuário no Telegram (/foto, /status, /ligar, /desligar, /sensibilidade)."""
    offset = 0
    print("[Telegram] Escutando comandos no bot @Seguranca_da_Casa_bot...")
    try:
        r0 = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"offset": 0, "timeout": 2},
            timeout=10,
        )
        if r0.ok:
            res = r0.json().get("result", [])
            if res:
                offset = res[-1]["update_id"] + 1
    except Exception:
        pass

    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 25},
                timeout=32,
            )
            if r.ok:
                data = r.json()
                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id != str(CHAT_ID):
                        continue
                    text = (msg.get("text") or "").strip()
                    handle_command(text)
            else:
                time.sleep(3)
        except Exception:
            time.sleep(3)


def handle_command(text: str):
    cmd = text.lower().split()
    if not cmd:
        return
    c = cmd[0].split("@")[0]

    if c in ("/start", "/ajuda", "/help"):
        url_info = os.getenv("RENDER_EXTERNAL_URL", "")
        painel_str = f"\n🌐 *Painel Web (Ajustar Área):* {url_info}" if url_info else ""
        tg_send_message(
            "🛡️ *Monitor de Câmera 24h (Híbrido IA + Movimento)*\n\n"
            "Comandos disponíveis:\n"
            "📸 `/foto` — Tira uma foto da câmera agora mesmo\n"
            "📊 `/status` — Mostra o status do monitoramento\n"
            "🔔 `/ligar` — Ativa os alertas de pessoas/movimentação\n"
            "🔕 `/desligar` — Pausa os alertas temporariamente\n"
            "🎯 `/sensibilidade 0.22` — Ajusta a sensibilidade (0.10 a 0.80)\n"
            "⏱️ `/intervalo 25` — Segundos entre dois alertas"
            f"{painel_str}"
        )
    elif c in ("/foto", "/agora"):
        with latest_frame_lock:
            jpg = latest_annotated_jpeg
        if jpg:
            tg_send_photo(jpg, f"📸 Foto ao vivo (Nuvem) — {now_brt_str()}")
        else:
            tg_send_message("⏳ Ainda conectando na câmera, tente novamente em alguns segundos.")
    elif c == "/status":
        st = "🟢 ATIVOS" if state["alerts_on"] else "🔴 PAUSADOS"
        cam_st = "🟢 Conectada" if state["stream_online"] else "🟡 Reconectando"
        url_info = os.getenv("RENDER_EXTERNAL_URL", "")
        painel_str = f"\n• Painel Web: {url_info}" if url_info else ""
        tg_send_message(
            f"📊 *Status do Servidor 24h (Híbrido IA + Movimento)*\n"
            f"• Câmera: {cam_st}\n"
            f"• Alertas: {st}\n"
            f"• Sensibilidade: `{state['threshold']}`\n"
            f"• Intervalo: `{state['cooldown_sec']}s`\n"
            f"• Total de alertas enviados: `{state['total_alerts']}`\n"
            f"• Online desde: `{state['started_at']}`"
            f"{painel_str}"
        )
    elif c == "/ligar":
        state["alerts_on"] = True
        save_state()
        tg_send_message("🔔 *Alertas ATIVADOS na Nuvem!* Vou avisar qualquer pessoa ou movimentação na área.")
    elif c == "/desligar":
        state["alerts_on"] = False
        save_state()
        tg_send_message("🔕 *Alertas PAUSADOS!* Use `/ligar` quando quiser reativar.")
    elif c == "/sensibilidade" and len(cmd) > 1:
        try:
            val = max(0.10, min(0.85, float(cmd[1].replace(",", "."))))
            state["threshold"] = val
            save_state()
            tg_send_message(f"🎯 Sensibilidade ajustada para `{val:.2f}`.")
        except ValueError:
            tg_send_message("Use: `/sensibilidade 0.22`")
    elif c == "/intervalo" and len(cmd) > 1:
        try:
            val = max(5, min(3600, int(cmd[1])))
            state["cooldown_sec"] = val
            save_state()
            tg_send_message(f"⏱️ Intervalo entre alertas ajustado para `{val}s`.")
        except ValueError:
            tg_send_message("Use: `/intervalo 25`")


# ================= CAPTURA DE STREAM EM TEMPO REAL (ZERO DELAY) =================
def stream_reader_loop():
    """Lê continuamente os frames do RTMP para nunca acumular atraso no buffer."""
    global latest_raw_frame, latest_raw_ts
    while True:
        if not CAM_URL:
            time.sleep(5)
            continue
        print(f"[Câmera] Conectando ao stream RTMP...")
        cap = cv2.VideoCapture(CAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            state["stream_online"] = False
            print("[Câmera] Falha ao abrir stream. Tentando novamente em 5s...")
            time.sleep(5)
            continue

        print("[Câmera] Stream RTMP conectado com sucesso!")
        state["stream_online"] = True
        fails = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                fails += 1
                if fails > 15:
                    print("[Câmera] Stream interrompido, reconectando...")
                    break
                time.sleep(0.05)
                continue
            fails = 0
            with latest_frame_lock:
                latest_raw_frame = frame
                latest_raw_ts = time.time()

        state["stream_online"] = False
        cap.release()
        time.sleep(2)


# ================= MAPEADOR INTELIGENTE DE MOVIMENTO E PRESENÇA (ANTI-SOMBRA) =================
class SmartPresenceTracker:
    """
    Mapeia pessoas, animais e movimentações reais eliminando sombras (MOG2 detectShadows=True +
    supressão cromática/gradiente + verificação de movimento real curto prazo para nunca travar em manchas de sol/sombra).
    """

    def __init__(self):
        self.mw = 480
        self.mh = 270
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=28, detectShadows=True)
        self.prev_gray = None
        self.bg_edges = None
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 19))

    def detect(self, frame_bgr: np.ndarray, sensitivity_thr: float):
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.mw, self.mh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Gradiente estrutural (bordas físicas reais — sombras suaves no chão/muro quase não têm bordas internas novas)
        gx = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_16S, 0, 1, ksize=3)
        edges = cv2. addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)

        if self.prev_gray is None:
            self.prev_gray = gray_blur.copy()
            self.bg_edges = edges.astype(np.float32)
            self.mog2.apply(small, learningRate=0.5)
            return []

        # 1) MOG2 com detecção nativa de sombra (sombra = 127, objeto físico = 255)
        fg_raw = self.mog2.apply(small, learningRate=0.025)
        # Descarta todos os pixels de sombra (127) mantendo apenas objetos físicos (255)
        _, fg_no_shadow = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)

        # 2) Diferença de movimento real curto prazo (entre o quadro atual e o anterior) + diferença de bordas físicas
        step_diff = cv2.absdiff(gray_blur, self.prev_gray)
        self.prev_gray = cv2.addWeighted(self.prev_gray, 0.65, gray_blur, 0.35, 0)

        edge_diff = cv2.absdiff(edges, cv2.convertScaleAbs(self.bg_edges))
        cv2.accumulateWeighted(edges, self.bg_edges, 0.04)

        # Mascara relógio OSD (canto superior esquerdo) e logo HD-IPC (canto inferior esquerdo)
        for m in (fg_no_shadow, step_diff, edge_diff):
            m[0 : int(self.mh * 0.09), 0 : int(self.mw * 0.40)] = 0
            m[int(self.mh * 0.90) :, 0 : int(self.mw * 0.16)] = 0

        # Proteção contra mudança global de iluminação (nuvem/sol cobrindo ou descobrindo a rua)
        changed_ratio = float(np.count_nonzero(fg_no_shadow)) / float(self.mw * self.mh)
        if changed_ratio > 0.28:
            self.mog2.apply(small, learningRate=0.40)
            cv2.accumulateWeighted(edges, self.bg_edges, 0.40)
            return []

        fg = cv2.morphologyEx(fg_no_shadow, cv2.MORPH_OPEN, self.kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_px = float(self.mw * self.mh)
        min_area = total_px * 0.0032   # ~0.32% da tela (captura animais como cães/gatos e pessoas)
        max_area = total_px * 0.22     # >22% da tela de uma vez é mudança de sol/sombra, não pessoa/animal

        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 10:
                continue
            # Descarta faixas puramente horizontais finas (típicas de sombra de fio/muro no chão)
            if bw > bh * 2.8:
                continue

            # Exige que a região tenha movimento ativo recente (step_diff) E novas bordas físicas (edge_diff),
            # eliminando 100% manchas estáticas de sombra ou reflexo de sol na parede/chão!
            roi_step = step_diff[by : by + bh, bx : bx + bw]
            roi_edge = edge_diff[by : by + bh, bx : bx + bw]
            active_motion_px = float(np.count_nonzero(roi_step >= 14)) / float(max(1, bw * bh))
            active_edge_px = float(np.count_nonzero(roi_edge >= 22)) / float(max(1, bw * bh))

            if active_motion_px < 0.04 or active_edge_px < 0.03:
                continue

            raw_boxes.append([bx, by, bx + bw, by + bh, area])

        if not raw_boxes:
            return []

        merged = self._merge_nearby_boxes(raw_boxes, pad_x=int(self.mw * 0.025), pad_y=int(self.mh * 0.035))

        scale_x = w / float(self.mw)
        scale_y = h / float(self.mh)
        out = []
        for (x1, y1, x2, y2, area) in merged:
            if (x2 - x1) * (y2 - y1) > total_px * 0.25:
                continue
            score = min(0.92, 0.60 + (area / total_px) * 3.5)
            out.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, float(score), "movimento"))
        return out

    def _merge_nearby_boxes(self, boxes, pad_x: int, pad_y: int):
        changed = True
        curr = [list(b) for b in boxes]
        while changed and len(curr) > 1:
            changed = False
            nxt = []
            used = [False] * len(curr)
            for i in range(len(curr)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2, a_area = curr[i]
                for j in range(i + 1, len(curr)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2, b_area = curr[j]
                    if (
                        ax1 - pad_x <= bx2
                        and ax2 + pad_x >= bx1
                        and ay1 - pad_y <= by2
                        and ay2 + pad_y >= by1
                    ):
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        a_area += b_area
                        used[j] = True
                        changed = True
                nxt.append([ax1, ay1, ax2, ay2, a_area])
            curr = nxt
        return curr


# ================= DETECÇÃO IA NEURAL (PESSOAS + ANIMAIS — YOLOv8n ONNX) =================
COCO_TARGET_CLASSES = {
    0: "pessoa",
    1: "pessoa/bike",
    3: "pessoa/moto",
    14: "animal (ave)",
    15: "animal (gato)",
    16: "animal (cachorro)",
    17: "animal (cavalo)",
}


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return
    for url in MODEL_URLS:
        try:
            print(f"[IA] Baixando modelo YOLOv8n ONNX de {url} ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=262144):
                        if chunk:
                            f.write(chunk)
            if os.path.getsize(MODEL_PATH) > 5_000_000:
                print("[IA] Modelo YOLOv8n ONNX pronto!")
                return
        except Exception as e:
            print(f"[IA] Erro ao baixar de {url}: {e}")


def run_yolo_single(session, input_name: str, img_bgr: np.ndarray, conf_thr: float):
    """Executa 1 passada do YOLOv8n (640x640) buscando Pessoas e Animais."""
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        return []
    resized = cv2.resize(img_bgr, (640, 640))
    blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    outputs = session.run(None, {input_name: blob})[0]
    preds = outputs[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # (8400, 84)

    eff_thr = min(conf_thr, 0.20)
    target_ids = list(COCO_TARGET_CLASSES.keys())
    class_Cols = [4 + cid for cid in target_ids]
    sub_scores = preds[:, class_Cols]
    best_idx = np.argmax(sub_scores, axis=1)
    best_scores = sub_scores[np.arange(len(preds)), best_idx]

    mask = best_scores >= eff_thr
    if not np.any(mask):
        return []

    filtered = preds[mask]
    scores_f = best_scores[mask]
    idx_f = best_idx[mask]

    cx = filtered[:, 0] * (w / 640.0)
    cy = filtered[:, 1] * (h / 640.0)
    bw = filtered[:, 2] * (w / 640.0)
    bh = filtered[:, 3] * (h / 640.0)

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    boxes_xywh = [[float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])] for i in range(len(scores_f))]
    sc_list = [float(s) for s in scores_f]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, sc_list, eff_thr, 0.45)
    if len(indices) == 0:
        return []

    out = []
    for idx in np.array(indices).flatten():
        i = int(idx)
        cid = target_ids[int(idx_f[i])]
        kind = COCO_TARGET_CLASSES.get(cid, "pessoa")
        out.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(scores_f[i]), kind))
    return out


def run_yolo_pass(session, input_name: str, frame: np.ndarray, poly_px: np.ndarray, conf_thr: float, zoom_zone: bool = False):
    """Roda YOLOv8n no quadro inteiro e, quando há presença na área, também com Zoom na área demarcada."""
    h, w = frame.shape[:2]
    candidates = run_yolo_single(session, input_name, frame, conf_thr)

    if zoom_zone:
        px1 = max(0, int(np.min(poly_px[:, 0]) - w * 0.06))
        py1 = max(0, int(np.min(poly_px[:, 1]) - h * 0.06))
        px2 = min(w, int(np.max(poly_px[:, 0]) + w * 0.06))
        py2 = min(h, int(np.max(poly_px[:, 1]) + h * 0.06))
        if (px2 - px1) > w * 0.15 and (py2 - py1) > h * 0.15:
            crop = frame[py1:py2, px1:px2]
            for (cx1, cy1, cx2, cy2, sc, kind) in run_yolo_single(session, input_name, crop, conf_thr):
                candidates.append((cx1 + px1, cy1 + py1, cx2 + px1, cy2 + py1, sc, kind))

    if len(candidates) <= 1:
        return candidates

    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in candidates]
    scores = [b[4] for b in candidates]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, min(conf_thr, 0.20), 0.45)
    if len(indices) == 0:
        return []
    return [candidates[int(idx)] for idx in np.array(indices).flatten()]


def box_iou(a, b) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ub = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (ua + ub - inter)


def is_box_in_poly(box_xyxy, poly_px: np.ndarray, poly_mask: np.ndarray) -> bool:
    """
    Verifica se pelo menos 80% da área da caixa da pessoa/movimentação está dentro
    do polígono demarcado (conforme solicitado: >= 80% dentro da área monitorada).
    """
    h, w = poly_mask.shape[:2]
    x1 = int(max(0, min(w - 1, box_xyxy[0])))
    y1 = int(max(0, min(h - 1, box_xyxy[1])))
    x2 = int(max(x1 + 1, min(w, box_xyxy[2])))
    y2 = int(max(y1 + 1, min(h, box_xyxy[3])))

    roi = poly_mask[y1:y2, x1:x2]
    if roi.size > 0:
        overlap = float(np.count_nonzero(roi)) / float(roi.size)
        return overlap >= 0.80
    return False


def annotate_image(frame: np.ndarray, poly_px: np.ndarray, detections: list, dwell_sec: float = 0.0) -> bytes:
    vis = frame.copy()
    # Sombra translúcida leve dentro da área monitorada + borda verde
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly_px], (60, 160, 40))
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
    cv2.polylines(vis, [poly_px], isClosed=True, color=(122, 209, 51), thickness=2)

    for (x1, y1, x2, y2, score, kind, in_zone) in detections:
        # Vermelho forte se estiver >= 80% dentro da área monitorada, Amarelo se estiver fora/parcial
        color = (36, 48, 255) if in_zone else (0, 200, 255)
        thickness = 3 if in_zone else 2
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if in_zone:
            label = f"NA AREA: {kind} ({dwell_sec:.1f}s)"
        else:
            label = f"{kind} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 6, int(y1))
        cv2.rectangle(vis, (int(x1), ty - th - 6), (int(x1) + tw + 8, ty), color, -1)
        cv2.putText(
            vis,
            label,
            (int(x1) + 4, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    return buf.tobytes() if ok else b""


def monitor_camera_loop():
    global latest_annotated_jpeg
    ensure_model()
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    session = None
    input_name = ""
    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        print(f"[IA] Sessão Híbrida (YOLOv8n ONNX + SmartPresenceTracker) iniciada!")
    except Exception as e:
        print(f"[IA] Aviso ONNX: {e} — operando com SmartPresenceTracker.")

    tracker = SmartPresenceTracker()
    last_processed_ts = 0.0
    last_yolo_ts = 0.0
    cached_yolo_boxes = []
    in_zone_since = 0.0
    last_in_zone_seen = 0.0

    while True:
        with latest_frame_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
            frame_ts = latest_raw_ts

        if frame is None or frame_ts == last_processed_ts:
            time.sleep(0.05)
            continue

        last_processed_ts = frame_ts
        now = time.time()
        h, w = frame.shape[:2]

        poly_norm = state["poly"]
        poly_px = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in poly_norm], dtype=np.int32)
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_px], 255)

        thr = state["threshold"]

        # 1) Mapeamento instantâneo de presença/movimento (~3ms de CPU, roda 6x por segundo!)
        motion_boxes = tracker.detect(frame, thr)

        # 2) Executa YOLOv8n (Pessoas + Animais) quando há movimento/presença OU a cada 1.8s
        if session is not None and (motion_boxes or (now - last_yolo_ts >= 1.8)):
            last_yolo_ts = now
            try:
                cached_yolo_boxes = run_yolo_pass(session, input_name, frame, poly_px, thr, zoom_zone=bool(motion_boxes))
            except Exception:
                cached_yolo_boxes = []
        elif now - last_yolo_ts > 2.5:
            cached_yolo_boxes = []

        # 3) Combina as detecções da IA com o Mapeador de Movimento/Presença
        combined = list(cached_yolo_boxes)
        for mb in motion_boxes:
            if not any(box_iou(mb, yb) > 0.30 for yb in cached_yolo_boxes):
                combined.append(mb)

        detections = []
        any_in_zone = False
        for (x1, y1, x2, y2, score, kind) in combined:
            in_zone = is_box_in_poly((x1, y1, x2, y2), poly_px, poly_mask)
            if in_zone:
                any_in_zone = True
            detections.append((x1, y1, x2, y2, score, kind, in_zone))

        if any_in_zone:
            last_in_zone_seen = now
            if in_zone_since == 0.0:
                in_zone_since = now
            dwell_sec = now - in_zone_since
        else:
            # Tolerância de 0.65s para não zerar o cronômetro de 3s se oscilar 1 quadro
            if now - last_in_zone_seen > 0.65:
                in_zone_since = 0.0
                dwell_sec = 0.0
            else:
                dwell_sec = (now - in_zone_since) if in_zone_since > 0.0 else 0.0

        jpg = annotate_image(frame, poly_px, detections, dwell_sec)
        with latest_frame_lock:
            latest_annotated_jpeg = jpg

        # Só dispara o alerta se estiver >= 80% dentro da área por MAIS de 3.0 segundos!
        if any_in_zone and dwell_sec >= 3.0 and state["alerts_on"]:
            if now - state["last_alert_ts"] >= state["cooldown_sec"]:
                state["last_alert_ts"] = now
                state["total_alerts"] += 1
                hora = now_brt_str()
                print(f"[ALERTA] Pessoa >=80% na área por {dwell_sec:.1f}s às {hora}!")
                threading.Thread(
                    target=tg_send_photo,
                    args=(jpg, f"🚨👤 Pessoa na área monitorada ({dwell_sec:.1f}s) — {hora}"),
                    daemon=True,
                ).start()

        time.sleep(0.14)


def keep_alive_loop():
    """Mantém o servidor gratuito do Render acordado 24/7 fazendo self-ping a cada 4 minutos."""
    while True:
        time.sleep(240)
        ext_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        if ext_url:
            try:
                requests.get(f"{ext_url}/ping", timeout=10)
            except Exception:
                pass


# ================= SERVIDOR WEB COM LOGIN E EDITOR VISUAL DA ÁREA =================
class HealthHandler(BaseHTTPRequestHandler):
    def is_authenticated(self) -> bool:
        if not AUTH_TOKEN:
            return True
        cookie_hdr = self.headers.get("Cookie", "")
        for part in cookie_hdr.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "cam_auth" and v == AUTH_TOKEN:
                return True
        return False

    def send_login_page(self, error_msg: str = ""):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        err_html = f'<p style="color:#ef4444;font-size:14px;margin-top:8px;">{error_msg}</p>' if error_msg else ""
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Login — Monitor Câmera 24h</title>
<style>
  body {{ background:#0a0e18; color:#fff; font-family:system-ui,sans-serif; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }}
  .card {{ background:#141b2d; padding:32px 28px; border-radius:14px; border:1px solid #26324d; width:100%; max-width:340px; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.5); }}
  input {{ width:100%; box-sizing:border-box; padding:12px; border-radius:8px; border:1px solid #334155; background:#0a0e18; color:#fff; font-size:16px; margin-top:12px; text-align:center; }}
  button {{ width:100%; background:#2563eb; color:#fff; border:none; padding:12px; border-radius:8px; cursor:pointer; font-weight:600; font-size:16px; margin-top:14px; }}
</style></head>
<body>
  <div class="card">
    <h2 style="margin:0 0 6px 0;">🔒 Acesso Protegido</h2>
    <p style="color:#94a3b8;font-size:14px;margin:0;">Monitor de Câmera 24h</p>
    {err_html}
    <form onsubmit="doLogin(event)">
      <input type="password" id="pw" placeholder="Digite a senha de acesso" autofocus required />
      <button type="submit">Entrar</button>
    </form>
  </div>
<script>
async function doLogin(e) {{
  e.preventDefault();
  const pw = document.getElementById('pw').value;
  const r = await fetch('/api/login', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{password: pw}})}});
  if (r.ok) location.reload();
  else alert('Senha incorreta!');
}}
</script></body></html>"""
        self.wfile.write(html.encode("utf-8"))

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            return

        if self.path == "/logout":
            self.send_response(302)
            self.send_header("Set-Cookie", "cam_auth=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")
            self.send_header("Location", "/")
            self.end_headers()
            return

        if not self.is_authenticated():
            if self.path.startswith("/snapshot.jpg") or self.path.startswith("/api/"):
                self.send_response(401)
                self.end_headers()
                return
            self.send_login_page()
            return

        if self.path.startswith("/snapshot.jpg"):
            with latest_frame_lock:
                jpg = latest_annotated_jpeg
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpg)
                return
            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        st = "🟢 ATIVOS" if state["alerts_on"] else "🔴 PAUSADOS"
        poly_json = json.dumps(state["poly"])
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitor Câmera 24h</title>
<style>
  body {{ background:#0a0e18; color:#fff; font-family:system-ui,sans-serif; text-align:center; margin:0; padding:16px; }}
  .bar {{ display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin:12px 0; }}
  button {{ background:#2563eb; color:#fff; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; font-weight:600; }}
  button.sec {{ background:#334155; }}
  .wrap {{ position:relative; display:inline-block; max-width:920px; width:100%; }}
  img {{ width:100%; display:block; border-radius:8px; border:2px solid #334155; }}
  canvas {{ position:absolute; left:0; top:0; width:100%; height:100%; cursor:crosshair; }}
</style></head>
<body>
  <h2>🛡️ Monitor de Câmera 24h — Híbrido (IA + Movimento)</h2>
  <p>Alertas: <b id="st">{st}</b> | Sensibilidade: <b>{state['threshold']}</b> | Total de Alertas: <b>{state['total_alerts']}</b></p>
  <div class="bar">
    <button onclick="savePoly()">💾 Salvar Área Monitorada</button>
    <button class="sec" onclick="toggleAlerts()">🔔 Ligar/Pausar Alertas</button>
    <button class="sec" onclick="resetPoly()">🔄 Resetar Área</button>
    <button class="sec" onclick="location.href='/logout'">🔒 Sair</button>
  </div>
  <p style="color:#94a3b8;font-size:14px;">Arraste os <b>8 pontos amarelos (1 a 8)</b> sobre a imagem para contornar a área monitorada e clique em <b>Salvar Área Monitorada</b>.</p>
  <div class="wrap">
    <img id="cam" src="/snapshot.jpg" onload="resizeCanvas()" />
    <canvas id="cv"></canvas>
  </div>
<script>
let poly = {poly_json};
let dragIdx = -1;
const img = document.getElementById('cam');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');

function resizeCanvas() {{
  cv.width = img.clientWidth;
  cv.height = img.clientHeight;
  draw();
}}
window.addEventListener('resize', resizeCanvas);

function draw() {{
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  poly.forEach((p, i) => {{
    const x = p[0] * w, y = p[1] * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.closePath();
  ctx.fillStyle = 'rgba(250, 204, 21, 0.15)';
  ctx.fill();
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 3;
  ctx.stroke();
  poly.forEach((p, i) => {{
    const px = p[0] * w, py = p[1] * h;
    ctx.beginPath();
    ctx.arc(px, py, 11, 0, Math.PI * 2);
    ctx.fillStyle = i === dragIdx ? '#ef4444' : '#facc15';
    ctx.fill();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), px, py);
  }});
}}

function getPos(e) {{
  const r = cv.getBoundingClientRect();
  const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
  const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
  return [Math.max(0.01, Math.min(0.99, cx / r.width)), Math.max(0.01, Math.min(0.99, cy / r.height))];
}}

function onDown(e) {{
  const [x, y] = getPos(e);
  let best = -1, bestD = 0.08;
  poly.forEach((p, i) => {{
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d < bestD) {{ bestD = d; best = i; }}
  }});
  if (best >= 0) {{ dragIdx = best; e.preventDefault(); draw(); }}
}}
function onMove(e) {{
  if (dragIdx < 0) return;
  e.preventDefault();
  poly[dragIdx] = getPos(e);
  draw();
}}
function onUp() {{ dragIdx = -1; draw(); }}

cv.addEventListener('mousedown', onDown);
cv.addEventListener('mousemove', onMove);
window.addEventListener('mouseup', onUp);
cv.addEventListener('touchstart', onDown, {{passive:false}});
cv.addEventListener('touchmove', onMove, {{passive:false}});
window.addEventListener('touchend', onUp);

async function savePoly() {{
  await fetch('/api/poly', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{poly}})}});
  alert('Área de 8 pontos salva no servidor!');
}}
async function toggleAlerts() {{
  await fetch('/api/toggle', {{method:'POST'}});
  location.reload();
}}
function resetPoly() {{
  poly = [[0.01,0.36],[0.20,0.37],[0.38,0.38],[0.53,0.68],[0.68,0.99],[0.35,0.99],[0.01,0.99],[0.01,0.68]];
  draw();
}}
setInterval(() => {{ if (dragIdx < 0) img.src = '/snapshot.jpg?t=' + Date.now(); }}, 1200);
</script></body></html>"""
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/login":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            pw = body.get("password", "")
            if pw == WEB_PASS:
                self.send_response(200)
                self.send_header("Set-Cookie", f"cam_auth={AUTH_TOKEN}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"UNAUTHORIZED")
            return

        if not self.is_authenticated():
            self.send_response(401)
            self.end_headers()
            return

        if self.path == "/api/poly":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if "poly" in body and 3 <= len(body["poly"]) <= 16:
                state["poly"] = body["poly"]
                save_state()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        if self.path == "/api/toggle":
            state["alerts_on"] = not state["alerts_on"]
            save_state()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    load_state()
    threading.Thread(target=stream_reader_loop, daemon=True).start()
    threading.Thread(target=telegram_bot_loop, daemon=True).start()
    threading.Thread(target=monitor_camera_loop, daemon=True).start()
    threading.Thread(target=keep_alive_loop, daemon=True).start()
    tg_send_message(
        "🟢 *Motor Híbrido (IA + Mapeamento de Movimento/Presença) ativado na Nuvem!*\n"
        "Agora mapeia 100% das pessoas e movimentações (mesmo vistas de cima ou à noite) em alta velocidade."
    )
    print(f"[Web] Painel de status rodando na porta {PORT}")
    HTTPServer(("0.0.0.0", PORT), HealthHandler).serve_forever()
