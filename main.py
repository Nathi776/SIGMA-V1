import cv2
import mediapipe as mp
import numpy as np
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.frame_buffer import FrameBuffer
from core.gait_database import GaitDatabase
from core.gait_utils import (
    normalize_pose,
    extract_gait_signature,
    extract_biomechanical_features,
    landmarks_to_bbox,
    bbox_center,
)
from core.person_tracker import SimplePersonTracker
from core.threaded_camera import ThreadedCamera


# -----------------------------
# Configuration
# -----------------------------
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"

LAPTOP_CAM_INDEX = 0
PHONE_CAM_URL = "http://172.16.12.79:4747/video"

MAX_SEQUENCE_FRAMES = 30
SIGNATURE_WINDOW = 5
MIN_VISIBILITY = 0.4
COSINE_MATCH_THRESHOLD = 0.15  # Calibrated: Intra < 0.07, Inter > 0.20


# -----------------------------
# Drawing Helpers
# -----------------------------
def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, x, y, color=(0, 255, 0), bg_color=(20, 20, 20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x, max(0, y - h - 6)),
        (x + w + 4, y + baseline),
        bg_color,
        -1
    )
    cv2.putText(
        frame,
        text,
        (x + 2, y - 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def pose_is_usable(landmarks):
    visible_count = 0
    for lm in landmarks:
        visibility = getattr(lm, "visibility", 1.0)
        if visibility >= MIN_VISIBILITY:
            visible_count += 1
    return visible_count >= 20


# -----------------------------
# Camera Pipeline
# -----------------------------
class CameraPipeline:
    """
    Dedicated pipeline for a single camera stream.
    
    Includes:
    - Threaded asynchronous video capture (non-blocking, zero-latency buffer)
    - Dedicated MediaPipe PoseLandmarker and FaceDetector instances
    - Strictly increasing per-camera monotonic timestamps
    - Dedicated person tracker and gait sequence buffers
    - Local track to global SIGMA identity mapping
    """

    def __init__(self, cam_id, name, source, pose_model_path, face_model_path=None):
        self.cam_id = cam_id
        self.name = name
        self.source = source

        # Threaded capture
        self.camera = ThreadedCamera(source=source, name=name)

        # Isolated MediaPipe PoseLandmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=4
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Isolated MediaPipe FaceDetector
        self.face_detector = None
        if face_model_path and os.path.exists(face_model_path):
            face_options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=face_model_path),
                running_mode=vision.RunningMode.VIDEO
            )
            self.face_detector = vision.FaceDetector.create_from_options(face_options)

        # Spatial person tracking
        self.tracker = SimplePersonTracker(max_distance=120, max_missing=20)

        # Buffers: track_id -> FrameBuffer
        self.person_buffers = {}

        # Identity assignments: track_id -> dict(person_id, distance, is_new, status)
        self.track_identities = {}

        # Strictly monotonic timestamp generator
        self.start_time = time.monotonic()
        self.last_timestamp_ms = 0

    def get_monotonic_timestamp_ms(self):
        """Guarantees strictly increasing timestamp (now > last) required by MediaPipe."""
        now_ms = int((time.monotonic() - self.start_time) * 1000)
        if now_ms <= self.last_timestamp_ms:
            now_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now_ms
        return now_ms

    def process_person_gait(self, track_id, landmarks, database):
        gait_vector = normalize_pose(landmarks)

        if track_id not in self.person_buffers:
            self.person_buffers[track_id] = FrameBuffer(
                max_frames=MAX_SEQUENCE_FRAMES,
                signature_window=SIGNATURE_WINDOW
            )

        buffer = self.person_buffers[track_id]
        buffer.add(gait_vector, landmarks)

        # Buffer not yet filled with enough frames
        if not buffer.is_full():
            return {
                "status": "BUFFERING",
                "progress": f"{len(buffer.buffer)}/{MAX_SEQUENCE_FRAMES}",
                "person_id": None,
                "distance": None,
                "is_new": False
            }

        sequence = buffer.get_sequence()
        landmark_seq = buffer.get_landmarks()

        base_signature = extract_gait_signature(sequence)
        if base_signature is None:
            return None

        bio_features = extract_biomechanical_features(landmark_seq)
        signature = np.concatenate([base_signature, bio_features]).astype(np.float32)

        buffer.add_signature(signature)
        stable_signature = buffer.get_stable_signature(min_samples=3)

        if stable_signature is None:
            return {
                "status": "BUFFERING",
                "progress": f"Stabilizing {len(buffer.signature_buffer)}/3",
                "person_id": None,
                "distance": None,
                "is_new": False
            }

        # If this track already has an identity assigned, update smoothly with EMA
        if track_id in self.track_identities:
            assigned_id = self.track_identities[track_id]["person_id"]
            current_dist = database.get_distance(assigned_id, stable_signature)
            database.update_signature(assigned_id, stable_signature, alpha=0.05)
            self.track_identities[track_id]["distance"] = current_dist
            return {
                "status": "MATCHED",
                "person_id": assigned_id,
                "distance": current_dist,
                "is_new": False
            }

        # Otherwise, match against known identities or auto-enroll
        person_id, distance, is_new = database.match(stable_signature, auto_enroll=True)
        self.track_identities[track_id] = {
            "person_id": person_id,
            "distance": distance,
            "is_new": is_new,
        }

        return {
            "status": "NEW" if is_new else "MATCHED",
            "person_id": person_id,
            "distance": distance,
            "is_new": is_new
        }

    def release(self):
        self.camera.release()
        try:
            self.pose_landmarker.close()
            if self.face_detector is not None:
                self.face_detector.close()
        except Exception:
            pass


# -----------------------------
# Main Application
# -----------------------------
def main():
    database = GaitDatabase(
        db_path="data/gait/gait_db.json",
        threshold=COSINE_MATCH_THRESHOLD
    )

    print("=" * 65)
    print("SIGMA-V1 MULTI-CAMERA INTELLIGENCE & GAIT RE-IDENTIFICATION")
    print("=" * 65)
    print(f"[INFO] Loaded {len(database.signatures)} enrolled identity profile(s):")
    for pid in sorted(database.signatures.keys()):
        name = database.get_name(pid)
        obs = database.profiles.get(pid, {}).get("observations", 1)
        print(f"  - SIGMA ID {pid}: '{name}' ({obs} observations)")
    print("-" * 65)
    print("Controls:")
    print("  'q' - Quit and save database")
    print("  'v' - Toggle unified side-by-side dashboard / separate windows")
    print("  'c' - Clear / reset gait database")
    print("=" * 65)

    pipelines = [
        CameraPipeline(
            cam_id=0,
            name="Camera 1 (Laptop)",
            source=LAPTOP_CAM_INDEX,
            pose_model_path=POSE_MODEL_PATH,
            face_model_path=FACE_MODEL_PATH,
        ),
        CameraPipeline(
            cam_id=1,
            name="Camera 2 (Phone)",
            source=PHONE_CAM_URL,
            pose_model_path=POSE_MODEL_PATH,
            face_model_path=FACE_MODEL_PATH,
        ),
    ]

    use_unified_dashboard = True

    try:
        while True:
            processed_frames = []
            active_pids_per_camera = {}

            for pipeline in pipelines:
                success, frame = pipeline.camera.read()

                if not success or frame is None:
                    # Render friendly placeholder if stream is disconnected or reconnecting
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    status_text = f"{pipeline.name}: Waiting for stream..."
                    draw_label(placeholder, status_text, 30, 230, (0, 165, 255))
                    draw_label(placeholder, f"Source: {pipeline.source}", 30, 265, (180, 180, 180))
                    processed_frames.append(placeholder)
                    active_pids_per_camera[pipeline.cam_id] = set()
                    continue

                frame_h, frame_w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = pipeline.get_monotonic_timestamp_ms()

                # Process Pose and Face independently on this camera's timeline
                pose_result = pipeline.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                face_result = (
                    pipeline.face_detector.detect_for_video(mp_image, timestamp_ms)
                    if pipeline.face_detector is not None
                    else None
                )

                # Filter usable pose detections
                detections = []
                if pose_result.pose_landmarks:
                    for landmarks in pose_result.pose_landmarks:
                        if not pose_is_usable(landmarks):
                            continue

                        bbox = landmarks_to_bbox(landmarks, frame_w, frame_h, padding=25)
                        center = bbox_center(bbox)
                        detections.append({
                            "bbox": bbox,
                            "center": center,
                            "landmarks": landmarks
                        })

                # Update spatial tracker
                tracks = pipeline.tracker.update(detections)

                # Clean up stale buffers and identities for dead tracks
                active_track_ids = set(tracks.keys())
                dead_buffer_ids = [tid for tid in pipeline.person_buffers if tid not in active_track_ids]
                for tid in dead_buffer_ids:
                    del pipeline.person_buffers[tid]

                dead_identity_ids = [tid for tid in pipeline.track_identities if tid not in active_track_ids]
                for tid in dead_identity_ids:
                    del pipeline.track_identities[tid]

                cam_active_pids = set()

                # Process each tracked person
                for track_id, track in tracks.items():
                    if track["missing"] > 0:
                        continue

                    bbox = track["bbox"]
                    landmarks = track["landmarks"]
                    x1, y1, x2, y2 = bbox

                    result = pipeline.process_person_gait(track_id, landmarks, database)

                    # Draw person bounding box
                    draw_bbox(frame, bbox, color=(0, 255, 255), thickness=2)
                    draw_label(frame, f"Track {track_id}", x1, max(25, y1 - 32), color=(0, 255, 255))

                    if result is not None:
                        status = result["status"]
                        if status == "BUFFERING":
                            draw_label(
                                frame,
                                f"Gait: {result['progress']}",
                                x1,
                                max(50, y1 - 8),
                                color=(0, 200, 255)
                            )
                        elif status in ("MATCHED", "NEW"):
                            pid = result["person_id"]
                            dist = result["distance"]
                            cam_active_pids.add(pid)
                            name = database.get_name(pid)
                            color = (0, 255, 0) if status == "MATCHED" else (0, 140, 255)
                            dist_str = f"d={dist:.3f}" if dist is not None else "d=0.0"
                            draw_label(
                                frame,
                                f"ID {pid}: {name} | {status} ({dist_str})",
                                x1,
                                max(50, y1 - 8),
                                color=color
                            )

                active_pids_per_camera[pipeline.cam_id] = cam_active_pids

                # Draw Face Bounding Boxes
                if face_result and face_result.detections:
                    for detection in face_result.detections:
                        bbox = detection.bounding_box
                        fx, fy, fw, fh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 120, 0), 2)
                        draw_label(frame, "FACE", fx, max(20, fy - 8), color=(255, 120, 0))

                # HUD Overlay
                fps = pipeline.camera.fps
                hud_text = (
                    f"{pipeline.name} | FPS: {fps:.1f} | Tracks: {len(tracks)} | "
                    f"DB IDs: {len(database.signatures)}"
                )
                draw_label(frame, hud_text, 15, 28, color=(0, 255, 0), bg_color=(20, 20, 20))

                processed_frames.append(frame)

            # Check cross-camera identity association
            pids_c1 = active_pids_per_camera.get(0, set())
            pids_c2 = active_pids_per_camera.get(1, set())
            common_pids = pids_c1.intersection(pids_c2)

            # Display options: Unified Side-by-Side or Separate Windows
            if use_unified_dashboard and len(processed_frames) == 2:
                # Resize both to standard 640x480 for clean horizontal stacking
                f1 = cv2.resize(processed_frames[0], (640, 480))
                f2 = cv2.resize(processed_frames[1], (640, 480))
                dashboard = np.hstack([f1, f2])

                # Bottom intelligence status banner
                bar = np.zeros((45, 1280, 3), dtype=np.uint8)
                if common_pids:
                    matched_names = ", ".join([f"ID {p} ({database.get_name(p)})" for p in common_pids])
                    banner = f"*** CROSS-CAMERA MATCH: {matched_names} ACTIVE ON BOTH CAMERAS! ***"
                    draw_label(bar, banner, 20, 30, color=(0, 255, 0), bg_color=(10, 50, 10))
                else:
                    status_msg = (
                        f"SIGMA-V1 ACTIVE | Enrolled: {len(database.signatures)} | "
                        f"[v]: Toggle View | [c]: Reset DB | [q]: Quit"
                    )
                    draw_label(bar, status_msg, 20, 30, color=(200, 200, 200), bg_color=(30, 30, 30))

                full_view = np.vstack([dashboard, bar])
                cv2.imshow("SIGMA-V1 Multi-Camera Intelligence", full_view)
            else:
                for idx, pframe in enumerate(processed_frames):
                    cv2.imshow(pipelines[idx].name, pframe)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting application.")
                break
            elif key == ord("v"):
                use_unified_dashboard = not use_unified_dashboard
                cv2.destroyAllWindows()
                print(f"[INFO] Unified dashboard mode: {use_unified_dashboard}")
            elif key == ord("c"):
                print("[INFO] Resetting GaitDatabase.")
                database.clear()

    finally:
        print("[INFO] Cleaning up resources...")
        database.save()
        for pipeline in pipelines:
            pipeline.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")


if __name__ == "__main__":
    main()