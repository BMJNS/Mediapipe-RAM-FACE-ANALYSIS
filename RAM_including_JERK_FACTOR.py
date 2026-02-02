import cv2
from collections import deque

import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
print("MediaPipe version:", mp.__version__)

# ----------------- Paths / Config -----------------
BASE_DIR = "/Users/bmjonas04/Desktop/ACABI/hand_test"
BASELINE_IMAGE_PATH = f"{BASE_DIR}/photo jerky.jpg"
VIDEO_PATH = f"{BASE_DIR}/video jerky.mov"

MAX_NUM_HANDS = 1
SMOOTH_WINDOW = 10

# Outlier rules
OUTLIER_K = 2.0              # excursion-based filter (EXISTING)
DURATION_OUTLIER_K = 2.0     # duration-based filter (NEW)

# Jerkiness (mid-phase only, ignore turning points)
JERK_P_LOW = 0.2
JERK_P_HIGH = 0.8

# ----------------- MediaPipe setup -----------------
mp_hands = mp.solutions.hands

# ----------------- Helper functions -----------------
def distance_px(lm1, lm2, img_shape):
    h, w, _ = img_shape
    x1, y1 = lm1.x * w, lm1.y * h
    x2, y2 = lm2.x * w, lm2.y * h
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def compute_amplitude(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    return distance_px(lm[0], lm[12], img_shape)

def simple_moving_average(values, window):
    """
    Centered moving average (offline).
    """
    n = len(values)
    if window <= 1 or n == 0:
        return values[:]

    left = window // 2
    right = window - left - 1

    out = []
    for i in range(n):
        start = max(0, i - left)
        end = min(n - 1, i + right)
        seg = values[start:end + 1]
        out.append(sum(seg) / len(seg))
    return out

# ----------------- Jerkiness helpers (NEW) -----------------
def finite_diff_velocity(times, values):
    """Compute velocity using forward differences."""
    v = []
    for i in range(len(values) - 1):
        dt = times[i + 1] - times[i]
        if dt <= 0:
            v.append(0.0)
        else:
            v.append((values[i + 1] - values[i]) / dt)
    return v

def speed_cv(values):
    """Coefficient of variation = std/mean (dimensionless)."""
    if len(values) < 3:
        return 0.0
    mean_v = float(np.mean(values))
    if mean_v == 0.0:
        return 0.0
    return float(np.std(values, ddof=1) / mean_v)

def midphase_mask_by_progress(amps_segment, p_low=0.2, p_high=0.8):
    """
    Return indices (within the segment) that fall in the mid-phase based on normalized progress p.
    p = (a - a_min) / (a_max - a_min) -> p in [0,1]
    """
    a_min = float(np.min(amps_segment))
    a_max = float(np.max(amps_segment))
    E = a_max - a_min
    if E <= 1e-6:
        return []  # no movement

    keep = []
    for i, a in enumerate(amps_segment):
        p = (a - a_min) / E
        if p_low <= p <= p_high:
            keep.append(i)
    return keep

def jerkiness_speed_cv_midphase(times, amps, i0, i1, p_low=0.2, p_high=0.8):
    """
    Compute jerkiness for a phase amps[i0:i1+1] by:
    1) keeping only mid-phase frames via progress p in [p_low, p_high]
    2) computing velocity on the kept samples
    3) returning CV of speed = std(|v|)/mean(|v|)
    """
    if i1 <= i0:
        return 0.0

    t_seg = times[i0:i1 + 1]
    a_seg = amps[i0:i1 + 1]

    keep_idx = midphase_mask_by_progress(a_seg, p_low=p_low, p_high=p_high)
    if len(keep_idx) < 4:
        return 0.0

    t_keep = [t_seg[i] for i in keep_idx]
    a_keep = [a_seg[i] for i in keep_idx]

    v = finite_diff_velocity(t_keep, a_keep)
    speeds = [abs(x) for x in v]
    return speed_cv(speeds)

def compute_cycle_jerkiness(cycle, times, amps, p_low=0.2, p_high=0.8):
    """
    Compute jerkiness for a full cycle by averaging mid-phase speed-CV
    across the two halves: (max->min) and (min->max).
    """
    s = cycle["start_i"]
    m = cycle["mid_i"]
    e = cycle["end_i"]

    j1 = jerkiness_speed_cv_midphase(times, amps, s, m, p_low=p_low, p_high=p_high)  # open->close
    j2 = jerkiness_speed_cv_midphase(times, amps, m, e, p_low=p_low, p_high=p_high)  # close->open
    return 0.5 * (j1 + j2)

# ----------------- Cycle detection -----------------
def detect_cycles(times, amps, baseline_amp, smooth_window=10):
    if len(times) < 5:
        return []

    smoothed = simple_moving_average(amps, smooth_window)
    diffs = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]

    maxima_idx, minima_idx = [], []
    for i in range(1, len(diffs)):
        if diffs[i - 1] > 0 and diffs[i] <= 0:
            maxima_idx.append(i)
        elif diffs[i - 1] < 0 and diffs[i] >= 0:
            minima_idx.append(i)

    events = sorted(
        [(i, "max") for i in maxima_idx] + [(i, "min") for i in minima_idx],
        key=lambda x: x[0]
    )

    cycles = []
    i = 0
    while i < len(events) - 2:
        idx1, t1 = events[i]
        idx2, t2 = events[i + 1]
        idx3, t3 = events[i + 2]

        if t1 == "max" and t2 == "min" and t3 == "max":
            start_i, end_i = idx1, idx3
            start_time = times[start_i]
            end_time = times[end_i]
            duration = end_time - start_time

            segment = amps[start_i:end_i + 1]
            peak_open = max(segment)
            excursion_pct = ((max(segment) - min(segment)) / baseline_amp) * 100.0

            cycles.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "amplitude": peak_open,
                "excursion_pct": excursion_pct,
                "is_outlier": False,
                "outlier_reason": "",

                # NEW: indices for mid-phase jerkiness
                "start_i": start_i,
                "mid_i": idx2,
                "end_i": end_i,

                # NEW: filled later
                "jerkiness": 0.0,
            })
            i += 2
        else:
            i += 1

    return cycles

# ----------------- Outlier filters -----------------
def flag_small_excursion_outliers(cycles, k=2.0):
    if len(cycles) < 4:
        return cycles, []

    excs = [c["excursion_pct"] for c in cycles]
    kept, outliers = [], []

    for i, c in enumerate(cycles):
        others = excs[:i] + excs[i + 1:]
        mean_o = np.mean(others)
        std_o = np.std(others, ddof=1) if len(others) > 1 else 0.0
        thresh = mean_o - k * std_o

        bad = c["excursion_pct"] < thresh if std_o > 0 else c["excursion_pct"] < mean_o
        if bad:
            c["is_outlier"] = True
            c["outlier_reason"] = (
                f"excursion {c['excursion_pct']:.2f}% < "
                f"LOO threshold {thresh:.2f}%"
            )
            outliers.append(c)
        else:
            kept.append(c)

    return kept, outliers

def flag_short_duration_outliers(cycles, k=2.0):
    if len(cycles) < 4:
        return cycles, []

    durs = [c["duration"] for c in cycles]
    kept, outliers = [], []

    for i, c in enumerate(cycles):
        others = durs[:i] + durs[i + 1:]
        mean_o = np.mean(others)
        std_o = np.std(others, ddof=1) if len(others) > 1 else 0.0
        thresh = mean_o - k * std_o

        bad = c["duration"] < thresh if std_o > 0 else c["duration"] < mean_o
        if bad:
            c["is_outlier"] = True
            if c["outlier_reason"]:
                c["outlier_reason"] += " + short duration"
            else:
                c["outlier_reason"] = (
                    f"duration {c['duration']:.3f}s < "
                    f"LOO threshold {thresh:.3f}s"
                )
            outliers.append(c)
        else:
            kept.append(c)

    return kept, outliers

# ----------------- Baseline & video -----------------
def compute_baseline_amplitude(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read baseline image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        raise RuntimeError("No hand detected in baseline image.")
    return compute_amplitude(result.multi_hand_landmarks[0], img.shape)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    times, amplitudes = [], []

    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                times.append(t)
                amplitudes.append(
                    compute_amplitude(result.multi_hand_landmarks[0], frame.shape)
                )

    cap.release()
    return times, amplitudes

# ----------------- Main -----------------
def main():
    baseline_amp = compute_baseline_amplitude(BASELINE_IMAGE_PATH)
    times, amplitudes = process_video(VIDEO_PATH)

    if len(times) == 0:
        print("No hand data found in the video.")
        return

    t0 = times[0]
    norm_times = [t - t0 for t in times]

    cycles_all = detect_cycles(norm_times, amplitudes, baseline_amp, SMOOTH_WINDOW)

    if len(cycles_all) == 0:
        print("No clear open–close–open cycles detected.")
        return

    # Existing excursion-based filter
    cycles, out_amp = flag_small_excursion_outliers(cycles_all, k=OUTLIER_K)

    # Duration-based filter (applied AFTER excursion filter)
    cycles, out_dur = flag_short_duration_outliers(cycles, k=DURATION_OUTLIER_K)

    outlier_cycles = out_amp + out_dur

    # NEW: jerkiness computed on kept cycles (ignores turning points via p in [0.2, 0.8])
    for c in cycles:
        c["jerkiness"] = compute_cycle_jerkiness(
            c, norm_times, amplitudes, p_low=JERK_P_LOW, p_high=JERK_P_HIGH
        )

    # ----------------- Text output -----------------
    print("\n===== RAM Hand Test Summary (Offline Video) =====")
    print(f"Baseline fully-extended amplitude (from photo): {baseline_amp:.1f} px")
    print(f"Detected cycles (raw): {len(cycles_all)}")
    print(f"Kept cycles (after filters): {len(cycles)}")
    print(
        f"Outlier cycles removed: {len(outlier_cycles)} "
        f"(excursion k={OUTLIER_K}, duration k={DURATION_OUTLIER_K})"
    )
    print(f"Jerkiness window: mid-phase only (p in [{JERK_P_LOW}, {JERK_P_HIGH}])")

    if outlier_cycles:
        print("\nOutliers (removed from totals/graphs):")
        for j, c in enumerate(outlier_cycles, start=1):
            print(
                f"  Outlier {j:2d}: "
                f"start={c['start_time']:5.2f}s, end={c['end_time']:5.2f}s, "
                f"duration={c['duration']:.3f}s, excursion={c['excursion_pct']:.1f}%  "
                f"-> {c['outlier_reason']}"
            )

    if len(cycles) == 0:
        print("\nAll cycles were flagged as outliers.")
        return

    # ----------------- Main metrics (kept cycles only) -----------------
    durations = [c["duration"] for c in cycles]
    cycle_amplitudes = [c["amplitude"] for c in cycles]
    amp_percent = [(a / baseline_amp) * 100.0 for a in cycle_amplitudes]
    cycle_indices = np.arange(1, len(cycles) + 1)
    cycle_mid_times = [(c["start_time"] + c["end_time"]) / 2.0 for c in cycles]

    avg_duration = float(np.mean(durations))
    avg_amp = float(np.mean(cycle_amplitudes))

    fastest = min(durations)
    slowest = max(durations)
    max_amp_cycle = max(cycle_amplitudes)
    min_amp_cycle = min(cycle_amplitudes)

    # NEW jerkiness summary
    jerk_vals = [c["jerkiness"] for c in cycles]
    avg_jerk = float(np.mean(jerk_vals))
    max_jerk = float(np.max(jerk_vals))
    min_jerk = float(np.min(jerk_vals))

    if len(cycle_indices) >= 2:
        speed_m, speed_b = np.polyfit(cycle_mid_times, durations, 1)
        amp_pct_m, amp_pct_b = np.polyfit(cycle_indices, amp_percent, 1)
        jerk_m, jerk_b = np.polyfit(cycle_mid_times, jerk_vals, 1)  # jerkiness trend
    else:
        speed_m = speed_b = amp_pct_m = amp_pct_b = jerk_m = jerk_b = 0.0

    # ----------------- Summary -----------------
    print("\n===== Summary (Kept Cycles Only) =====")
    print(f"Completed cycles: {len(cycles)}")
    print(f"Fastest cycle duration: {fastest:.3f} s")
    print(f"Slowest cycle duration: {slowest:.3f} s")
    print(f"Largest movement amplitude (cycle): {max_amp_cycle:.1f} px")
    print(f"Smallest movement amplitude (cycle): {min_amp_cycle:.1f} px")
    print(f"Average cycle duration (speed): {avg_duration:.3f} s")
    print(f"Average cycle amplitude: {avg_amp:.1f} px")

    print(f"\nJerkiness (mid-phase speed CV): avg={avg_jerk:.3f}, min={min_jerk:.3f}, max={max_jerk:.3f}")

    print("\nTrend (best-fit line slopes):")
    print(f"  Speed slope vs time: {speed_m:.4f} sec/s")
    print(f"  Amplitude % slope vs cycle index: {amp_pct_m:.4f} %/cycle")
    print(f"  Jerkiness slope vs time: {jerk_m:.4f} /s  (+ = getting jerkier over time)")

    print("\nPer-cycle details (kept cycles):")
    for i, c in enumerate(cycles, start=1):
        pct_open = (c["amplitude"] / baseline_amp) * 100.0
        print(
            f"  Cycle {i:2d}: "
            f"start={c['start_time']:5.2f}s, end={c['end_time']:5.2f}s, "
            f"duration={c['duration']:.3f}s, "
            f"amplitude={c['amplitude']:.1f}px ({pct_open:.1f}% of baseline), "
            f"open–close diff={c['excursion_pct']:.1f}%, "
            f"jerkiness={c['jerkiness']:.3f}"
        )

    # ----------------- Plots -----------------
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_indices, amp_percent, "o-", label="Cycle amplitude (% of baseline)")
    plt.axhline(100.0, linestyle="--", label="Baseline (100%)")
    if len(cycle_indices) >= 2:
        plt.plot(cycle_indices, amp_pct_m * cycle_indices + amp_pct_b, "-", label="Amplitude % trend")
    plt.xlabel("Cycle index")
    plt.ylabel("Amplitude (% of fully extended baseline)")
    plt.title("Cycle Amplitude vs Baseline (Outliers Removed)")
    plt.ylim(35, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(cycle_mid_times, durations, "o-", label="Cycle duration")
    if len(cycle_mid_times) >= 2:
        t_arr = np.array(cycle_mid_times)
        plt.plot(t_arr, speed_m * t_arr + speed_b, "-", label="Speed trend")
    plt.xlabel("Time (s from start of video)")
    plt.ylabel("Cycle duration (s)")
    plt.title("Cycle Speed Over Time (Outliers Removed)")
    plt.ylim(0.2, 2.0)  # changed from (0.25, 1.3)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # NEW: Jerkiness over time (kept cycles only)
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_mid_times, jerk_vals, "o-", label="Jerkiness (mid-phase speed CV)")
    if len(cycle_mid_times) >= 2:
        t_arr = np.array(cycle_mid_times)
        plt.plot(t_arr, jerk_m * t_arr + jerk_b, "-", label="Jerkiness trend")
    plt.xlabel("Time (s from start of video)")
    plt.ylabel("Jerkiness (CV of speed)")
    plt.title("Jerkiness Over Time (Turning Points Ignored)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
