import cv2
from collections import deque

import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# ----------------- Paths / Config -----------------
BASE_DIR = "/Users/bmjonas04/Desktop/ACABI/hand_test"
BASELINE_IMAGE_PATH = f"{BASE_DIR}/photo 3.jpg"
VIDEO_PATH = f"{BASE_DIR}/video 3.mov"

# used for hand number and also how many windows for the moving average

MAX_NUM_HANDS = 1
SMOOTH_WINDOW = 10

# ----------------- MediaPipe setup -----------------
mp_hands = mp.solutions.hands

# ----------------- Helper functions -----------------
def distance_px(lm1, lm2, img_shape):
    """Euclidean distance between two normalized landmarks in pixel space."""
    h, w, _ = img_shape
    x1, y1 = lm1.x * w, lm1.y * h
    x2, y2 = lm2.x * w, lm2.y * h
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def compute_amplitude(hand_landmarks, img_shape):
    """
    Amplitude metric for one frame:
    distance between middle fingertip (landmark 12) and wrist (landmark 0).
    """
    lm = hand_landmarks.landmark
    wrist = lm[0]
    middle_tip = lm[12]
    return distance_px(wrist, middle_tip, img_shape)

def simple_moving_average(values, window):
    '''
    Return a list of values for the moving avergaes of the amplitudes during the video
    based on the window size.

    :param values: List of amplitudes
    :param window: Window size of moving average
    :return:
    '''

    # if window too small or we have less values than the window just return copy
    if window <= 1 or len(values) <= window:
        return values[:]

    # cumsum will be sum of nums for current "window" using sliding window of adding new and removing old
    out = []
    cumsum = 0.0
    for i, v in enumerate(values):

        # add new val to window
        cumsum += v

        # with new added val if total num bigger then window remove oldest
        if i >= window:
            cumsum -= values[i - window]


        if i >= window - 1:
            out.append(cumsum / window)

        # if at beginning and dont have full window yet
        else:
            out.append(cumsum / (i + 1))
    return out

def detect_cycles(times, amps, smooth_window=10):
    """
    Detect open→close→open cycles using local max/min of amplitude signal.
    A cycle = max (open) -> min (closed) -> max (open).
    Returns list of dicts with start_time, end_time, duration, amplitude.
    """
    if len(times) < 5:
        return []

    smoothed = simple_moving_average(amps, smooth_window)

    # first derivative
    diffs = [smoothed[i+1] - smoothed[i] for i in range(len(smoothed)-1)]

    maxima_idx = []
    minima_idx = []

    for i in range(1, len(diffs)):
        prev = diffs[i-1]
        curr = diffs[i]
        if prev > 0 and curr <= 0:
            maxima_idx.append(i)
        elif prev < 0 and curr >= 0:
            minima_idx.append(i)

    events = sorted(
        [(idx, "max") for idx in maxima_idx] +
        [(idx, "min") for idx in minima_idx],
        key=lambda x: x[0]
    )

    cycles = []
    i = 0
    while i < len(events) - 2:
        idx1, t1 = events[i]
        idx2, t2 = events[i+1]
        idx3, t3 = events[i+2]

        if t1 == "max" and t2 == "min" and t3 == "max":
            start_i = idx1
            end_i = idx3
            if start_i < len(times) and end_i < len(times):
                start_time = times[start_i]
                end_time = times[end_i]
                duration = end_time - start_time
                cycle_amp = max(amps[start_i:end_i+1])
                cycles.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "amplitude": cycle_amp,
                })
            i += 2
        else:
            i += 1

    return cycles

# ----------------- Baseline from still image -----------------
def compute_baseline_amplitude(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read baseline image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as hands_img:
        result = hands_img.process(img_rgb)

    if not result.multi_hand_landmarks:
        raise RuntimeError("No hand detected in baseline image. "
                           "Make sure the hand is fully extended and visible.")

    hand_landmarks = result.multi_hand_landmarks[0]
    baseline_amp = compute_amplitude(hand_landmarks, img.shape)
    return baseline_amp

# ----------------- Process pre-recorded video -----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    times = []
    amplitudes = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_vid:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # timestamps from video file (in ms -> s)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_vid.process(frame_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                amp = compute_amplitude(hand_landmarks, frame.shape)
                times.append(t)
                amplitudes.append(amp)

    cap.release()
    return times, amplitudes

# ----------------- Main analysis -----------------
def main():
    # 1) Baseline from still image
    print("Computing baseline (fully extended) amplitude from image...")
    baseline_amp = compute_baseline_amplitude(BASELINE_IMAGE_PATH)
    print(f"Baseline amplitude (photo): {baseline_amp:.1f} px")

    # 2) Process video to get amplitude signal over time
    print("Processing video for movement cycles...")
    times, amplitudes = process_video(VIDEO_PATH)

    if len(amplitudes) == 0:
        print("No hand data found in the video. "
              "Check that the hand is in frame and MediaPipe is detecting it.")
        return

    # Normalize times so they start at 0
    t0 = times[0]
    norm_times = [t - t0 for t in times]

    # 3) Detect cycles
    cycles = detect_cycles(norm_times, amplitudes, smooth_window=SMOOTH_WINDOW)

    if len(cycles) == 0:
        print("No clear open–close–open cycles detected. "
              "Try more distinct movements in the video.")
        return

    durations = [c["duration"] for c in cycles]          # cycle duration = speed metric
    cycle_amplitudes = [c["amplitude"] for c in cycles]  # absolute amplitude (px)
    amp_percent = [(a / baseline_amp) * 100.0 for a in cycle_amplitudes]
    cycle_indices = np.arange(1, len(cycles) + 1)

    # Use cycle mid-time for “time of cycle”
    cycle_mid_times = [
        (c["start_time"] + c["end_time"]) / 2.0 for c in cycles
    ]

    avg_duration = float(np.mean(durations))
    avg_amp = float(np.mean(cycle_amplitudes))

    fastest = min(durations)
    slowest = max(durations)
    max_amp_cycle = max(cycle_amplitudes)
    min_amp_cycle = min(cycle_amplitudes)

    # Linear fits for trends, if at least 2 cycles
    if len(cycle_indices) >= 2:
        # speed vs time trend
        speed_m, speed_b = np.polyfit(cycle_mid_times, durations, 1)
        # amplitude % vs cycle index trend
        amp_pct_m, amp_pct_b = np.polyfit(cycle_indices, amp_percent, 1)
    else:
        speed_m = speed_b = amp_pct_m = amp_pct_b = 0.0

    # ----------------- Text summary -----------------
    print("\n===== RAM Hand Test Summary (Offline Video) =====")
    print(f"Baseline fully-extended amplitude (from photo): {baseline_amp:.1f} px")
    print(f"Completed cycles: {len(cycles)}")
    print(f"Fastest cycle duration: {fastest:.3f} s")
    print(f"Slowest cycle duration: {slowest:.3f} s")
    print(f"Largest movement amplitude (cycle): {max_amp_cycle:.1f} px")
    print(f"Smallest movement amplitude (cycle): {min_amp_cycle:.1f} px")
    print(f"Average cycle duration (speed): {avg_duration:.3f} s")
    print(f"Average cycle amplitude: {avg_amp:.1f} px")

    print("\nTrend (best-fit line slopes):")
    print(f"  Speed slope vs time: {speed_m:.4f} sec/s  "
          f"(+ = cycles getting longer over time)")
    print(f"  Amplitude % slope vs cycle index: {amp_pct_m:.4f} %/cycle "
          f"(+ = relatively larger openings over cycles)")

    print("\nPer-cycle details:")
    for i, c in enumerate(cycles, start=1):
        start_t = c["start_time"]
        end_t = c["end_time"]
        dur = c["duration"]
        amp = c["amplitude"]
        pct = (amp / baseline_amp) * 100.0

        print(
            f"  Cycle {i:2d}: "
            f"start={start_t:5.2f}s, end={end_t:5.2f}s, "
            f"duration={dur:.3f}s, amplitude={amp:.1f}px "
            f"({pct:.1f}% of baseline)"
        )

    # ----------------- Plots -----------------
    # 1) Amplitude vs baseline: per-cycle amplitude as % of baseline
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_indices, amp_percent, "o-", label="Cycle amplitude (% of baseline)")
    plt.axhline(100.0, linestyle="--", label="Baseline (100%)")
    if len(cycle_indices) >= 2:
        plt.plot(
            cycle_indices,
            amp_pct_m * cycle_indices + amp_pct_b,
            "-",
            label="Amplitude % trend",
        )
    plt.xlabel("Cycle index")
    plt.ylabel("Amplitude (% of fully extended baseline)")
    plt.title("Cycle Amplitude vs Baseline")
    plt.ylim(35, 100)  # fixed y-axis for amplitude comparison
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 2) Speed over time: cycle duration vs cycle mid-time
    plt.figure(figsize=(10, 4))
    plt.plot(cycle_mid_times, durations, "o-", label="Cycle duration")
    if len(cycle_mid_times) >= 2:
        t_arr = np.array(cycle_mid_times)
        plt.plot(t_arr, speed_m * t_arr + speed_b, "-", label="Speed trend")
    plt.xlabel("Time (s from start of video)")
    plt.ylabel("Cycle duration (s)")
    plt.title("Cycle Speed Over Time")
    plt.ylim(0.25, 1.3)  # fixed y-axis for duration comparison
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
