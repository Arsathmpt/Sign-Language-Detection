import streamlit as st
import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSense · Hand Gesture AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
  --bg:#050508;--surface:#0D0D14;--surface2:#14141F;--border:#1E1E30;
  --accent:#7C3AED;--accent2:#06D6A0;--accent3:#FFD166;--text:#EEEEFF;--muted:#5A5A7A;
}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;background-color:var(--bg);color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
.hero{padding:3rem 4rem 2rem;background:linear-gradient(160deg,#050508 0%,#0D0814 50%,#080514 100%);border-bottom:1px solid var(--border);position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-120px;left:-80px;width:500px;height:500px;background:radial-gradient(circle,rgba(124,58,237,0.15) 0%,transparent 65%);pointer-events:none;}
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:0.68rem;letter-spacing:0.3em;color:var(--accent2);text-transform:uppercase;margin-bottom:0.8rem;}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,7vw,5.5rem);line-height:0.95;letter-spacing:0.02em;margin:0;}
.hero-title .highlight{color:var(--accent);}
.hero-sub{color:var(--muted);font-size:0.9rem;line-height:1.7;max-width:540px;margin-top:0.8rem;font-weight:300;}
.pill-row{display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:1.4rem;}
.pill{padding:0.25rem 0.9rem;border-radius:999px;font-size:0.68rem;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;}
.pill-v{border:1px solid var(--accent);color:var(--accent);background:rgba(124,58,237,0.08);}
.pill-g{border:1px solid var(--accent2);color:var(--accent2);background:rgba(6,214,160,0.08);}
.pill-n{border:1px solid var(--border);color:var(--muted);background:var(--surface);}
.panel-label{font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.22em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;}
.gesture-display{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:2rem;text-align:center;margin-bottom:1.2rem;position:relative;overflow:hidden;}
.gesture-display::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));}
.gesture-emoji{font-size:4rem;line-height:1;margin-bottom:0.5rem;}
.gesture-name{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;letter-spacing:0.05em;color:var(--text);line-height:1;}
.gesture-conf{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--accent2);letter-spacing:0.15em;margin-top:0.4rem;}
.gesture-none{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--muted);letter-spacing:0.1em;}
.stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1.2rem;}
.stat-box{background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:1rem;text-align:center;}
.stat-val{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;color:var(--accent);line-height:1;}
.stat-key{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;margin-top:0.2rem;}
.history-item{display:flex;align-items:center;gap:0.8rem;padding:0.6rem 1rem;background:var(--surface2);border:1px solid var(--border);border-radius:10px;margin-bottom:0.4rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);}
.history-sign{color:var(--text);font-weight:500;}
.history-dot{width:6px;height:6px;border-radius:50%;background:var(--accent2);flex-shrink:0;}
.stButton>button{background:linear-gradient(135deg,var(--accent),#9F5FF1)!important;color:white!important;border:none!important;border-radius:12px!important;font-family:'Outfit',sans-serif!important;font-weight:600!important;font-size:0.9rem!important;padding:0.7rem 1.5rem!important;width:100%!important;letter-spacing:0.03em!important;box-shadow:0 4px 20px rgba(124,58,237,0.3)!important;}
.stButton>button:hover{opacity:0.88!important;}
div[data-testid="stImage"] img{border-radius:16px;width:100%;}
.stSuccess{background:rgba(6,214,160,0.08)!important;border:1px solid rgba(6,214,160,0.25)!important;border-radius:10px!important;color:var(--accent2)!important;}
.stInfo{background:rgba(124,58,237,0.07)!important;border:1px solid rgba(124,58,237,0.2)!important;border-radius:10px!important;color:#A78BFA!important;}
.stWarning{background:rgba(255,209,102,0.08)!important;border:1px solid rgba(255,209,102,0.25)!important;border-radius:10px!important;color:var(--accent3)!important;}
hr{border-color:var(--border)!important;}
.footer{border-top:1px solid var(--border);padding:1.2rem 4rem;display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;}
</style>
""", unsafe_allow_html=True)


# ── Model download + load ─────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"

@st.cache_resource(show_spinner=False)
def load_detector():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# ── Gesture classification ────────────────────────────────────────────────────
def finger_states(lm):
    t = lm[4].x < lm[3].x
    i = lm[8].y  < lm[6].y
    m = lm[12].y < lm[10].y
    r = lm[16].y < lm[14].y
    p = lm[20].y < lm[18].y
    return t, i, m, r, p

def dist(lm, a, b):
    return np.sqrt((lm[a].x-lm[b].x)**2+(lm[a].y-lm[b].y)**2)

def classify(lm):
    t,i,m,r,p = finger_states(lm)
    if dist(lm,4,8)<0.06 and m and r and p:   return "👌 OK", 0.91
    if t and i and not m and not r and p:       return "🤟 I Love You", 0.94
    if t and not i and not m and not r and p:   return "🤙 Call Me", 0.92
    if t and not i and not m and not r and not p:
        return ("👍 Thumbs Up",0.95) if lm[4].y<lm[9].y else ("👎 Thumbs Down",0.93)
    if not t and i and m and not r and not p:   return "✌️ Peace", 0.96
    if not t and i and not m and not r and not p: return "☝️ One", 0.97
    if t and i and m and r and p:
        return ("🖐️ Five",0.93) if dist(lm,4,20)>0.35 else ("✋ Open Hand",0.90)
    if not t and not i and not m and not r and not p: return "✊ Fist", 0.96
    return "❓ Unknown", 0.50

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
FINGERTIPS = {4,8,12,16,20}

def draw_hand(frame, lm):
    h,w = frame.shape[:2]
    for a,b in CONNECTIONS:
        cv2.line(frame,(int(lm[a].x*w),int(lm[a].y*h)),(int(lm[b].x*w),int(lm[b].y*h)),(124,58,237),2,cv2.LINE_AA)
    for idx,pt in enumerate(lm):
        cx,cy=int(pt.x*w),int(pt.y*h)
        cv2.circle(frame,(cx,cy),7 if idx in FINGERTIPS else 4,
                   (6,214,160) if idx in FINGERTIPS else (180,130,255),-1,cv2.LINE_AA)


# ── Session state ─────────────────────────────────────────────────────────────
for k,v in [("running",False),("history",deque(maxlen=8)),("total",0),("start",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="eyebrow">// Computer Vision · Real-Time · MediaPipe Tasks API</div>
  <div class="hero-title">Sign<span class="highlight">Sense</span></div>
  <div class="hero-sub">Real-time hand gesture recognition powered by MediaPipe Hand Landmarker.
  Classifies 9 ASL-inspired gestures instantly from your webcam — no GPU needed.</div>
  <div class="pill-row">
    <span class="pill pill-v">MediaPipe Tasks</span>
    <span class="pill pill-g">21 Landmarks</span>
    <span class="pill pill-n">9 Gestures</span>
    <span class="pill pill-n">Real-Time</span>
    <span class="pill pill-n">CPU Only</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_cam, col_info = st.columns([3,2], gap="large")

with col_cam:
    st.markdown("")
    st.markdown('<div class="panel-label">01 · Live Camera Feed</div>', unsafe_allow_html=True)
    b1,b2 = st.columns(2)
    with b1: start = st.button("▶  Start Detection")
    with b2: stop  = st.button("⏹  Stop")
    if start:
        st.session_state.running=True; st.session_state.start=time.time()
        st.session_state.total=0; st.session_state.history=deque(maxlen=8)
    if stop: st.session_state.running=False
    cam_frame  = st.empty()
    status_box = st.empty()

with col_info:
    st.markdown("")
    st.markdown('<div class="panel-label">02 · Detected Gesture</div>', unsafe_allow_html=True)
    gesture_box = st.empty()
    st.markdown('<div class="panel-label">03 · Session Stats</div>', unsafe_allow_html=True)
    stats_box   = st.empty()
    st.markdown('<div class="panel-label">04 · Detection Log</div>', unsafe_allow_html=True)
    history_box = st.empty()

# ── Render helpers ────────────────────────────────────────────────────────────
def render_gesture(name="", conf=0.0):
    if not name:
        gesture_box.markdown('<div class="gesture-display"><div class="gesture-emoji">🤚</div><div class="gesture-none">AWAITING HAND…</div></div>', unsafe_allow_html=True)
    else:
        parts=name.split(" ",1); emoji=parts[0]; label=parts[1] if len(parts)>1 else name
        gesture_box.markdown(f'<div class="gesture-display"><div class="gesture-emoji">{emoji}</div><div class="gesture-name">{label}</div><div class="gesture-conf">CONFIDENCE · {conf:.0%}</div></div>', unsafe_allow_html=True)

def render_stats(det=0,elapsed=0,fps=0):
    stats_box.markdown(f'<div class="stat-row"><div class="stat-box"><div class="stat-val">{det}</div><div class="stat-key">Detected</div></div><div class="stat-box"><div class="stat-val">{int(elapsed)}s</div><div class="stat-key">Session</div></div><div class="stat-box"><div class="stat-val">{fps}</div><div class="stat-key">FPS</div></div></div>', unsafe_allow_html=True)

def render_history():
    if not st.session_state.history:
        history_box.markdown('<div style="color:#5A5A7A;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;letter-spacing:0.1em;padding:0.5rem 0;">No detections yet…</div>', unsafe_allow_html=True)
        return
    html="".join(f'<div class="history-item"><div class="history-dot"></div><span class="history-sign">{e["sign"]}</span><span style="margin-left:auto;">{e["time"]}</span></div>' for e in reversed(list(st.session_state.history)))
    history_box.markdown(html, unsafe_allow_html=True)

render_gesture(); render_stats(); render_history()

# ── Detection loop ────────────────────────────────────────────────────────────
if st.session_state.running:
    with st.spinner("Loading model (first run downloads ~6MB)…"):
        detector = load_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_box.error("❌ Cannot access webcam. Check permissions.")
        st.session_state.running = False
    else:
        status_box.success("✓ Camera active — show your hand!")
        last_sign=""; last_sign_time=0
        frame_count=0; fps_start=time.time(); fps_val=0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            frame_count += 1
            now = time.time()

            if now - fps_start >= 1.0:
                fps_val = int(frame_count/(now-fps_start))
                frame_count=0; fps_start=now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            current_gesture=""; current_conf=0.0

            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    draw_hand(frame, hand_lm)
                    g, c = classify(hand_lm)
                    if c > current_conf:
                        current_gesture=g; current_conf=c

                if current_gesture and current_gesture != "❓ Unknown":
                    if current_gesture != last_sign or (now-last_sign_time) > 1.5:
                        st.session_state.total += 1
                        st.session_state.history.append({"sign":current_gesture,"time":time.strftime("%H:%M:%S")})
                        last_sign=current_gesture; last_sign_time=now

            # HUD overlay
            cv2.rectangle(frame,(0,0),(frame.shape[1],52),(5,5,8),-1)
            if current_gesture:
                cv2.putText(frame,current_gesture,(12,36),cv2.FONT_HERSHEY_SIMPLEX,0.9,(124,58,237),2,cv2.LINE_AA)
            cv2.putText(frame,f"FPS:{fps_val}",(frame.shape[1]-80,36),cv2.FONT_HERSHEY_SIMPLEX,0.6,(6,214,160),1,cv2.LINE_AA)

            cam_frame.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),channels="RGB",use_column_width=True)
            elapsed = now-(st.session_state.start or now)
            render_gesture(current_gesture,current_conf)
            render_stats(st.session_state.total,elapsed,fps_val)
            render_history()

        cap.release()
        status_box.info("⏹ Detection stopped.")
        render_gesture()

else:
    cam_frame.markdown("""
    <div style="height:380px;display:flex;flex-direction:column;align-items:center;justify-content:center;
         border:1px dashed #1E1E30;border-radius:16px;color:#5A5A7A;gap:1rem;background:#0D0D14;">
      <div style="font-size:3.5rem;">🤟</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;letter-spacing:0.18em;text-transform:uppercase;">
        Press Start to begin</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <span>SIGNSENSE · Built by Mohamed Arsath</span>
  <span>MediaPipe Tasks · OpenCV · Streamlit</span>
</div>""", unsafe_allow_html=True)
