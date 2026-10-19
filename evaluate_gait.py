import cv2
import os
import time
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.frame_buffer import FrameBuffer
from core.gait_database import GaitDatabase
from core.gait_utils import (
    normalize_pose,
    extract_gait_signature,
    extract_biomechanical_features,
    normalize_signature,
    landmarks_to_bbox,
    bbox_center,
)
from core.person_tracker import SimplePersonTracker


POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
BENCHMARK_DIR = "data/benchmark"
MAX_SEQUENCE_FRAMES = 30
SIGNATURE_WINDOW = 5
MIN_VISIBILITY = 0.4


def pose_is_usable(landmarks):
    visible = sum(1 for lm in landmarks if getattr(lm, "visibility", 1.0) >= MIN_VISIBILITY)
    return visible >= 20


def extract_signature_from_video(video_path, pose_model_path=POSE_MODEL_PATH):
    """
    Processes a recorded video clip and extracts the average representative gait signature.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    buffer = FrameBuffer(max_frames=MAX_SEQUENCE_FRAMES, signature_window=SIGNATURE_WINDOW)
    extracted_signatures = []

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            if pose_is_usable(landmarks):
                gait_vec = normalize_pose(landmarks)
                buffer.add(gait_vec, landmarks)

                if buffer.is_full():
                    seq = buffer.get_sequence()
                    lms = buffer.get_landmarks()
                    base_sig = extract_gait_signature(seq)
                    if base_sig is not None:
                        bio_feat = extract_biomechanical_features(lms)
                        sig = np.concatenate([base_sig, bio_feat]).astype(np.float32)
                        buffer.add_signature(sig)
                        stable_sig = buffer.get_stable_signature(min_samples=2)
                        if stable_sig is not None:
                            extracted_signatures.append(normalize_signature(stable_sig))

    cap.release()
    landmarker.close()

    if not extracted_signatures:
        print(f"[WARN] No stable gait signatures extracted from {os.path.basename(video_path)}")
        return None

    # Mean of all stable signatures extracted during the walking clip
    final_sig = np.mean(np.array(extracted_signatures), axis=0)
    return normalize_signature(final_sig)


def record_clip(person_name, pass_num, source=0, duration=6):
    """
    Records a walking pass of a person to data/benchmark/<Person>_pass<Num>.mp4.
    """
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    out_path = os.path.join(BENCHMARK_DIR, f"{person_name}_pass{pass_num}.mp4")

    # If source is digit string, convert to int
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {source}")
        return None

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"\n[RECORDING] Preparing to record '{person_name}' - Pass {pass_num}")
    print(f"File will be saved to: {out_path}")
    print("Get in position! Starting 3-second countdown...")

    # Countdown
    start_countdown = time.monotonic()
    while time.monotonic() - start_countdown < 3.0:
        ret, frame = cap.read()
        if not ret:
            continue
        remaining = 3 - int(time.monotonic() - start_countdown)
        cv2.putText(
            frame,
            f"GET READY: {remaining}",
            (int(width / 2) - 150, int(height / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 165, 255),
            3,
            cv2.LINE_AA
        )
        cv2.imshow("Recording Benchmark Clip", frame)
        cv2.waitKey(1)

    print(f"[RECORDING] >>> WALK NOW! Recording for {duration} seconds...")
    start_record = time.monotonic()
    frames_recorded = 0

    while time.monotonic() - start_record < duration:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frames_recorded += 1

        elapsed = time.monotonic() - start_record
        remaining = duration - elapsed
        cv2.putText(
            frame,
            f"REC [{person_name} Pass {pass_num}]: {remaining:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Recording Benchmark Clip", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[DONE] Saved {frames_recorded} frames to {out_path}\n")
    return out_path


def evaluate_benchmark(benchmark_dir=BENCHMARK_DIR, threshold=0.15):
    """
    Evaluates all recorded clips in benchmark_dir and computes intra-person vs inter-person distance matrix.
    """
    if not os.path.exists(benchmark_dir):
        print(f"[ERROR] Benchmark directory '{benchmark_dir}' not found.")
        return

    video_files = [f for f in os.listdir(benchmark_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    if len(video_files) < 2:
        print("[INFO] At least 2 recorded video clips are needed for cross-validation.")
        print(f"Found {len(video_files)} in {benchmark_dir}.")
        print("Use --record to record passes first, for example:")
        print("  python evaluate_gait.py --record --person Alice --pass 1")
        print("  python evaluate_gait.py --record --person Alice --pass 2")
        print("  python evaluate_gait.py --record --person Bob --pass 1")
        return

    print("=" * 65)
    print("EXTRACTING GAIT SIGNATURES FROM BENCHMARK CLIPS")
    print("=" * 65)

    signatures = {}
    for vf in sorted(video_files):
        vpath = os.path.join(benchmark_dir, vf)
        label = os.path.splitext(vf)[0]
        print(f"Processing: {vf} ...")
        sig = extract_signature_from_video(vpath)
        if sig is not None:
            signatures[label] = sig
            print(f"  -> Extracted 419-D signature (Norm: {np.linalg.norm(sig):.2f})")

    labels = list(signatures.keys())
    n = len(labels)
    if n < 2:
        print("[ERROR] Insufficient usable clips extracted.")
        return

    # Compute pairwise cosine distance matrix
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            u = signatures[labels[i]]
            v = signatures[labels[j]]
            dot = float(np.dot(u, v))
            dist_matrix[i, j] = max(0.0, 1.0 - dot)

    print("\n" + "=" * 65)
    print("PAIRWISE GAIT COSINE DISTANCE MATRIX (0.0 = Identical, 2.0 = Opposite)")
    print("=" * 65)

    header = f"{'Clip':<22}" + "".join([f"{l[:10]:>12}" for l in labels])
    print(header)
    print("-" * len(header))

    for i in range(n):
        row_str = f"{labels[i]:<22}"
        for j in range(n):
            row_str += f"{dist_matrix[i, j]:>12.4f}"
        print(row_str)

    # Classify into Intra-person (same person, different pass) vs Inter-person (different people)
    intra_distances = []
    inter_distances = []

    for i in range(n):
        for j in range(i + 1, n):
            person_i = labels[i].split("_pass")[0]
            person_j = labels[j].split("_pass")[0]
            dist = dist_matrix[i, j]

            if person_i == person_j:
                intra_distances.append((labels[i], labels[j], dist))
            else:
                inter_distances.append((labels[i], labels[j], dist))

    print("\n" + "=" * 65)
    print(f"IDENTITY MATCHING VALIDATION REPORT (Threshold = {threshold:.2f})")
    print("=" * 65)

    if intra_distances:
        print(f"SAME PERSON COMPARISONS (Intra-Class - Goal: < {threshold:.2f}):")
        for l1, l2, d in intra_distances:
            status = "MATCH [PASS]" if d < threshold else "MISMATCH [FAIL]"
            print(f"  {l1} <-> {l2} : dist = {d:.4f}  [{status}]")
        intra_vals = [d for _, _, d in intra_distances]
        print(f"  >> Average Intra-Person Distance: {np.mean(intra_vals):.4f}")
    else:
        print("[NOTICE] No repeated passes for the same person found yet (e.g. Alice_pass1 and Alice_pass2).")

    if inter_distances:
        print(f"\nDIFFERENT PERSON COMPARISONS (Inter-Class - Goal: >= {threshold:.2f}):")
        for l1, l2, d in inter_distances:
            status = "DISTINCT [PASS]" if d >= threshold else "FALSE MATCH [FAIL]"
            print(f"  {l1} <-> {l2} : dist = {d:.4f}  [{status}]")
        inter_vals = [d for _, _, d in inter_distances]
        print(f"  >> Average Inter-Person Distance: {np.mean(inter_vals):.4f}")

    if intra_distances and inter_distances:
        margin = np.mean(inter_vals) - np.mean(intra_vals)
        print(f"\n>> SEPARATION MARGIN (Inter - Intra): {margin:.4f}")
        if margin > 0.15:
            print("[SUCCESS] Strong discriminative margin! Gait recognition is reliably separating identities.")
        elif margin > 0.05:
            print("[ACCEPTABLE] Positive separation margin. Consider collecting more passes for calibration.")
        else:
            print("[WARN] Separation margin is narrow. Features or threshold should be tuned.")


def main():
    parser = argparse.ArgumentParser(description="SIGMA-V1 Gait Validation & Benchmarking Tool")
    parser.add_argument("--record", action="store_true", help="Record a walking pass video clip")
    parser.add_argument("--person", type=str, default="PersonA", help="Person name for the recording")
    parser.add_argument("--pass_num", type=int, default=1, help="Pass number (1, 2, 3...)")
    parser.add_argument("--camera", default=0, help="Camera index (0) or IP stream URL")
    parser.add_argument("--duration", type=int, default=6, help="Recording duration in seconds")
    parser.add_argument("--eval", action="store_true", help="Evaluate all recorded clips in data/benchmark")

    args = parser.parse_args()

    if args.record:
        record_clip(
            person_name=args.person,
            pass_num=args.pass_num,
            source=args.camera,
            duration=args.duration
        )
    elif args.eval:
        evaluate_benchmark()
    else:
        # Default behavior: run benchmark evaluation if clips exist, otherwise show help
        if os.path.exists(BENCHMARK_DIR) and len(os.listdir(BENCHMARK_DIR)) >= 2:
            evaluate_benchmark()
        else:
            parser.print_help()
            print("\nExample Quickstart:")
            print("  python evaluate_gait.py --record --person Prince --pass_num 1")
            print("  python evaluate_gait.py --record --person Prince --pass_num 2")
            print("  python evaluate_gait.py --eval")


if __name__ == "__main__":
    main()
