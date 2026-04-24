import time
from collections import deque
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="Dual-Lane Junction Controller", layout="wide")
st.title("Dual-Lane Junction Controller")
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1b2333 0%, #0d1117 55%, #090c10 100%);
        color: #e6edf3;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    .tcc-card-title {
        background: #161b22;
        color: #dbeafe;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 8px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        font-weight: 600;
    }
    .emergency-alert {
        margin-top: 8px;
        padding: 14px;
        border-radius: 12px;
        border: 2px solid #ef4444;
        background: rgba(239, 68, 68, 0.2);
        color: #fecaca;
        text-align: center;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: 0.5px;
        animation: tcc-blink 1s infinite;
    }
    @keyframes tcc-blink {
        0% { opacity: 1; box-shadow: 0 0 10px rgba(239, 68, 68, 0.7); }
        50% { opacity: 0.35; box-shadow: 0 0 0 rgba(239, 68, 68, 0.1); }
        100% { opacity: 1; box-shadow: 0 0 10px rgba(239, 68, 68, 0.7); }
    }
    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")
model_name = st.sidebar.text_input("YOLOv8 model", "yolov8n.pt")
hysteresis_buffer = st.sidebar.number_input("Hysteresis buffer (cars)", min_value=1, value=3, step=1)
min_green_seconds = st.sidebar.number_input("Minimum green time (seconds)", min_value=1, value=10, step=1)
lane_a_path = st.sidebar.text_input("Lane A video path", "lane_a.mp4.mp4")
lane_b_path = st.sidebar.text_input("Lane B video path", "lane_b.mp4.mp4")
surge_threshold = st.sidebar.number_input("Predictive surge threshold (cars)", min_value=1, value=5, step=1)
surge_window_seconds = st.sidebar.number_input("Trend window (seconds)", min_value=3, value=10, step=1)
surge_horizon_seconds = st.sidebar.number_input("Prediction horizon (seconds)", min_value=1, value=5, step=1)
max_red_seconds = 60

start_btn = st.sidebar.button("Start Monitoring")
stop_btn = st.sidebar.button("Stop")


if "running" not in st.session_state:
    st.session_state.running = False
if "model" not in st.session_state:
    st.session_state.model = None
if "green_lane" not in st.session_state:
    st.session_state.green_lane = "A"
if "cleared_a" not in st.session_state:
    st.session_state.cleared_a = 0
if "cleared_b" not in st.session_state:
    st.session_state.cleared_b = 0
if "green_since" not in st.session_state:
    st.session_state.green_since = time.time()
if "density_history_a" not in st.session_state:
    st.session_state.density_history_a = []
if "density_history_b" not in st.session_state:
    st.session_state.density_history_b = []
if "last_chart_update" not in st.session_state:
    st.session_state.last_chart_update = 0.0
if "emergency_since_a" not in st.session_state:
    st.session_state.emergency_since_a = None
if "emergency_since_b" not in st.session_state:
    st.session_state.emergency_since_b = None
if "flash_hist_a" not in st.session_state:
    st.session_state.flash_hist_a = deque(maxlen=10)
if "flash_hist_b" not in st.session_state:
    st.session_state.flash_hist_b = deque(maxlen=10)
if "density_timeline_a" not in st.session_state:
    st.session_state.density_timeline_a = deque(maxlen=300)
if "density_timeline_b" not in st.session_state:
    st.session_state.density_timeline_b = deque(maxlen=300)

if start_btn:
    st.session_state.running = True
    st.session_state.cleared_a = 0
    st.session_state.cleared_b = 0
    st.session_state.green_lane = "A"
    st.session_state.green_since = time.time()
    st.session_state.density_history_a = []
    st.session_state.density_history_b = []
    st.session_state.last_chart_update = 0.0
    st.session_state.emergency_since_a = None
    st.session_state.emergency_since_b = None
    st.session_state.flash_hist_a = deque(maxlen=10)
    st.session_state.flash_hist_b = deque(maxlen=10)
    st.session_state.density_timeline_a = deque(maxlen=300)
    st.session_state.density_timeline_b = deque(maxlen=300)
if stop_btn:
    st.session_state.running = False


lane_a_col, lane_b_col = st.columns(2)
lane_a_video = lane_a_col.empty()
lane_b_video = lane_b_col.empty()
lane_a_metrics = lane_a_col.empty()
lane_b_metrics = lane_b_col.empty()
lane_a_signal = lane_a_col.empty()
lane_b_signal = lane_b_col.empty()
lane_a_title = lane_a_col.empty()
lane_b_title = lane_b_col.empty()
junction_status = st.empty()
countdown_status = st.empty()
cleared_metrics = st.empty()
trend_status = st.empty()
trend_chart = st.empty()
emergency_alert = st.empty()
surge_alert = st.empty()


def has_flashing_lights(roi) -> bool:
    if roi is None or roi.size == 0:
        return False

    # Color-agnostic flashing check:
    # detect strong bright hotspots and high brightness contrast in the ROI.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    area = max(gray.shape[0] * gray.shape[1], 1)
    bright_ratio = (gray > 210).sum() / area
    contrast = float(gray.std())
    return bright_ratio > 0.015 and contrast > 35.0


def has_emergency_body_color(roi) -> bool:
    if roi is None or roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    area = max(roi.shape[0] * roi.shape[1], 1)

    # Red ranges in HSV (wraps around 0/180).
    red_1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_1, red_2) > 0

    # Yellow range in HSV.
    yellow_mask = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255)) > 0

    red_ratio = red_mask.sum() / area
    yellow_ratio = yellow_mask.sum() / area
    return red_ratio > 0.10 or yellow_ratio > 0.10


def count_and_annotate_cars(model: YOLO, frame):
    results = model(frame, verbose=False)[0]
    car_count = 0
    emergency_detected = False

    # COCO has no explicit "ambulance" class. We treat bus/truck as emergency candidates.
    emergency_vehicle_class_ids = {5, 7}

    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if class_id == 2:  # COCO class id for car
                car_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_id in emergency_vehicle_class_ids and confidence > 0.25:
                roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                body_color_match = has_emergency_body_color(roi)
                flashing = has_flashing_lights(roi)
                if body_color_match and flashing:
                    emergency_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        "EMERGENCY CANDIDATE",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return frame, car_count, emergency_detected


def render_lane_signal(column_placeholder, lane_name: str, is_green: bool):
    signal_text = "GREEN" if is_green else "RED"
    signal_color = "#22c55e" if is_green else "#ef4444"
    glow_color = "rgba(34,197,94,0.75)" if is_green else "rgba(239,68,68,0.75)"
    column_placeholder.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; gap:10px; margin-top:12px;">
            <div style="
                width:130px;
                height:130px;
                border-radius:50%;
                background:{signal_color};
                border:4px solid #f8fafc;
                box-shadow: 0 0 20px {glow_color}, 0 0 55px {glow_color};
            "></div>
            <div style="font-size:20px; font-weight:700; color:#e6edf3;">Lane {lane_name}: {signal_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_frame_loop(cap):
    ok, frame = cap.read()
    if ok:
        return True, frame

    # Rewind and try again so the lane feed loops forever.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap.read()


def predict_surge(timeline: deque, now_ts: float, window_sec: int, horizon_sec: int, threshold: int):
    while timeline and (now_ts - timeline[0][0]) > window_sec:
        timeline.popleft()

    if len(timeline) < 2:
        return False, 0.0

    start_t, start_count = timeline[0]
    end_t, end_count = timeline[-1]
    dt = max(end_t - start_t, 1e-6)
    growth_per_second = (end_count - start_count) / dt
    projected_increase = max(0.0, growth_per_second * horizon_sec)
    return projected_increase >= threshold, projected_increase


if st.session_state.running:
    if st.session_state.model is None:
        st.session_state.model = YOLO(model_name)

    cap_a = cv2.VideoCapture(lane_a_path)
    cap_b = cv2.VideoCapture(lane_b_path)

    if not cap_a.isOpened() or not cap_b.isOpened():
        st.error(
            "Could not open lane videos. Check paths in sidebar. "
            f"Lane A exists: {Path(lane_a_path).exists()} | Lane B exists: {Path(lane_b_path).exists()}"
        )
        st.stop()

    lane_a_title.markdown('<div class="tcc-card-title">Lane A Live Feed</div>', unsafe_allow_html=True)
    lane_b_title.markdown('<div class="tcc-card-title">Lane B Live Feed</div>', unsafe_allow_html=True)

    while st.session_state.running:
        ok_a, raw_frame_a = read_frame_loop(cap_a)
        ok_b, raw_frame_b = read_frame_loop(cap_b)

        if not ok_a or not ok_b:
            st.warning("Could not read one or both videos. Check file paths/codecs.")
            break

        frame_a, count_a, emergency_candidate_a = count_and_annotate_cars(st.session_state.model, raw_frame_a.copy())
        frame_b, count_b, emergency_candidate_b = count_and_annotate_cars(st.session_state.model, raw_frame_b.copy())

        now = time.time()
        st.session_state.density_timeline_a.append((now, count_a))
        st.session_state.density_timeline_b.append((now, count_b))

        surge_a, proj_a = predict_surge(
            st.session_state.density_timeline_a,
            now,
            int(surge_window_seconds),
            int(surge_horizon_seconds),
            int(surge_threshold),
        )
        surge_b, proj_b = predict_surge(
            st.session_state.density_timeline_b,
            now,
            int(surge_window_seconds),
            int(surge_horizon_seconds),
            int(surge_threshold),
        )

        st.session_state.flash_hist_a.append(emergency_candidate_a)
        st.session_state.flash_hist_b.append(emergency_candidate_b)

        # Requires repeated flashing-like detections over recent frames.
        stable_flash_a = sum(st.session_state.flash_hist_a) >= 2
        stable_flash_b = sum(st.session_state.flash_hist_b) >= 2

        if stable_flash_a:
            if st.session_state.emergency_since_a is None:
                st.session_state.emergency_since_a = now
        else:
            st.session_state.emergency_since_a = None

        if stable_flash_b:
            if st.session_state.emergency_since_b is None:
                st.session_state.emergency_since_b = now
        else:
            st.session_state.emergency_since_b = None

        confirmed_emergency_a = (
            st.session_state.emergency_since_a is not None
            and (now - st.session_state.emergency_since_a) >= 2.0
        )
        confirmed_emergency_b = (
            st.session_state.emergency_since_b is not None
            and (now - st.session_state.emergency_since_b) >= 2.0
        )

        emergency_lane = None
        if confirmed_emergency_a and not confirmed_emergency_b:
            emergency_lane = "A"
        elif confirmed_emergency_b and not confirmed_emergency_a:
            emergency_lane = "B"
        elif confirmed_emergency_a and confirmed_emergency_b:
            emergency_lane = "A" if count_a >= count_b else "B"

        if emergency_lane is not None:
            if st.session_state.green_lane != emergency_lane:
                st.session_state.green_lane = emergency_lane
                st.session_state.green_since = now
            emergency_alert.markdown(
                f'<div class="emergency-alert">EMERGENCY VEHICLE DETECTED - LANE {emergency_lane} PRIORITY GREEN</div>',
                unsafe_allow_html=True,
            )
        else:
            emergency_alert.empty()

        if surge_a or surge_b:
            messages = []
            if surge_a:
                messages.append(f"Lane A surge likely (+{proj_a:.1f} in next {int(surge_horizon_seconds)}s)")
            if surge_b:
                messages.append(f"Lane B surge likely (+{proj_b:.1f} in next {int(surge_horizon_seconds)}s)")
            surge_alert.warning("Predictive Surge Warning: " + " | ".join(messages))
        else:
            surge_alert.info("Predictive Surge Warning: No surge expected in near horizon.")

        elapsed_green = now - st.session_state.green_since
        remaining_lock = max(0.0, float(min_green_seconds) - elapsed_green)
        can_switch = remaining_lock <= 0
        red_duration_other_lane = elapsed_green
        seconds_to_max_red = max(0.0, max_red_seconds - red_duration_other_lane)

        # Restore adaptive logic:
        # 1) Normal switching follows min green time + hysteresis.
        # 2) Safety override forces switch if other lane has been red for too long.
        if emergency_lane is None:
            if red_duration_other_lane >= max_red_seconds:
                st.session_state.green_lane = "B" if st.session_state.green_lane == "A" else "A"
                st.session_state.green_since = now
                remaining_lock = float(min_green_seconds)
                seconds_to_max_red = float(max_red_seconds)
            elif st.session_state.green_lane == "A":
                if can_switch and count_b > count_a + hysteresis_buffer:
                    st.session_state.green_lane = "B"
                    st.session_state.green_since = now
                    remaining_lock = float(min_green_seconds)
                    seconds_to_max_red = float(max_red_seconds)
            else:
                if can_switch and count_a > count_b + hysteresis_buffer:
                    st.session_state.green_lane = "A"
                    st.session_state.green_since = now
                    remaining_lock = float(min_green_seconds)
                    seconds_to_max_red = float(max_red_seconds)

        if st.session_state.green_lane == "A":
            st.session_state.cleared_a += count_a
        else:
            st.session_state.cleared_b += count_b

        st.session_state.density_history_a.append(count_a)
        st.session_state.density_history_b.append(count_b)
        st.session_state.density_history_a = st.session_state.density_history_a[-120:]
        st.session_state.density_history_b = st.session_state.density_history_b[-120:]

        lane_a_video.image(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        lane_b_video.image(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        lane_a_metrics.metric("Lane A Cars", count_a)
        lane_b_metrics.metric("Lane B Cars", count_b)

        render_lane_signal(lane_a_signal, "A", st.session_state.green_lane == "A")
        render_lane_signal(lane_b_signal, "B", st.session_state.green_lane == "B")

        junction_status.info(
            f"Current GREEN Lane: {st.session_state.green_lane} | "
            f"Buffer: {hysteresis_buffer} cars | Min Green: {min_green_seconds}s | Max Red: {max_red_seconds}s | "
            f"Emergency Override: {'ON' if emergency_lane else 'OFF'}"
        )
        if emergency_lane is not None:
            countdown_status.warning("Emergency override active. Timer switch paused.")
        else:
            countdown_status.warning(
                f"AI unlock in {int(remaining_lock + 0.999)}s | "
                f"Max-red safety switch in {int(seconds_to_max_red + 0.999)}s"
            )
        cleared_metrics.metric(
            "Total Vehicles Cleared (A + B)",
            st.session_state.cleared_a + st.session_state.cleared_b,
            delta=f"A: {st.session_state.cleared_a} | B: {st.session_state.cleared_b}",
        )
        now_chart = time.time()
        if now_chart - st.session_state.last_chart_update >= 60:
            trend_status.markdown("### Vehicle Density Trend")
            trend_chart.line_chart(
                {
                    "Lane A": st.session_state.density_history_a,
                    "Lane B": st.session_state.density_history_b,
                },
                height=260,
            )
            st.session_state.last_chart_update = now_chart

        fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25
        fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25
        time.sleep(1 / max(fps_a, fps_b))

    cap_a.release()
    cap_b.release()
else:
    st.info("Set lane video paths in the sidebar, then click Start Monitoring.")
