import cv2
import mediapipe as mp
import numpy as np
import time

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


# -----------------------------
# Config
# -----------------------------
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
FACE_MODEL_PATH = "models/blaze_face_short_range.tflite"

LAPTOP_CAM_INDEX = 0
PHONE_CAM_URL = "http://192.168.53.21:4747/video"

MAX_SEQUENCE_FRAMES = 30
SIGNATURE_WINDOW = 5
MIN_VISIBILITY = 0.4


# -----------------------------
# Helpers
# -----------------------------
def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, x, y, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )


def pose_is_usable(landmarks):
    visible_count = 0
    for lm in landmarks:
        visibility = getattr(lm, "visibility", 1.0)
        if visibility >= MIN_VISIBILITY:
            visible_count += 1
    return visible_count >= 20


def process_person(track_id, landmarks, person_buffers, database):
    gait_vector = normalize_pose(landmarks)

    if track_id not in person_buffers:
        person_buffers[track_id] = FrameBuffer(
            max_frames=MAX_SEQUENCE_FRAMES,
            signature_window=SIGNATURE_WINDOW
        )

    buffer = person_buffers[track_id]
    buffer.add(gait_vector, landmarks)

    if not buffer.is_full():
        return None

    sequence = buffer.get_sequence()
    landmark_seq = buffer.get_landmarks()

    base_signature = extract_gait_signature(sequence)
    if base_signature is None:
        return None

    bio_features = extract_biomechanical_features(landmark_seq)
    signature = np.concatenate([base_signature, bio_features]).astype(np.float32)

    buffer.add_signature(signature)
    stable_signature = buffer.get_stable_signature()

    if stable_signature is None:
        return None

    person_id, distance, is_new = database.match(stable_signature)

    if not is_new:
        database.update_signature(person_id, stable_signature, alpha=0.1)

    return {
        "person_id": person_id,
        "distance": distance,
        "is_new": is_new
    }


# -----------------------------
# Initialize database
# -----------------------------
database = GaitDatabase(
    db_path="data/gait/gait_db.json",
    threshold=8.0
)


# -----------------------------
# Load MediaPipe models
# -----------------------------
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=4
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

face_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)
face_detector = vision.FaceDetector.create_from_options(face_options)


# -----------------------------
# Initialize cameras
# -----------------------------
cap_laptop = cv2.VideoCapture(LAPTOP_CAM_INDEX)
cap_phone = cv2.VideoCapture(PHONE_CAM_URL)

if not cap_laptop.isOpened():
    print("[ERROR] Could not open laptop camera.")
if not cap_phone.isOpened():
    print("[ERROR] Could not open phone camera.")

if not cap_laptop.isOpened() and not cap_phone.isOpened():
    raise RuntimeError("No cameras available.")


# -----------------------------
# Per-camera state
# -----------------------------
camera_sources = [
    {"name": "Camera 1", "cap": cap_laptop},
    {"name": "Camera 2", "cap": cap_phone},
]

camera_trackers = {
    0: SimplePersonTracker(max_distance=120, max_missing=20),
    1: SimplePersonTracker(max_distance=120, max_missing=20),
}

camera_person_buffers = {
    0: {},
    1: {},
}

start_time = time.monotonic()


# -----------------------------
# Main loop
# -----------------------------
while True:
    all_failed = True

    for cam_id, cam in enumerate(camera_sources):
        cap = cam["cap"]
        cam_name = cam["name"]

        if not cap.isOpened():
            continue

        ret, frame = cap.read()
        if not ret:
            draw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            draw_label(draw_frame, f"{cam_name}: no frame", 20, 40, (0, 0, 255))
            cv2.imshow(cam_name, draw_frame)
            continue

        all_failed = False
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

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

        tracks = camera_trackers[cam_id].update(detections)

        active_track_ids = set(tracks.keys())
        stale_buffer_ids = [
            tid for tid in camera_person_buffers[cam_id].keys()
            if tid not in active_track_ids
        ]
        for tid in stale_buffer_ids:
            del camera_person_buffers[cam_id][tid]

        for track_id, track in tracks.items():
            if track["missing"] > 0:
                continue

            bbox = track["bbox"]
            landmarks = track["landmarks"]

            result = process_person(
                track_id=track_id,
                landmarks=landmarks,
                person_buffers=camera_person_buffers[cam_id],
                database=database
            )

            draw_bbox(frame, bbox, (0, 255, 255), 2)

            x1, y1, x2, y2 = bbox
            draw_label(frame, f"Track {track_id}", x1, max(20, y1 - 30), (0, 255, 255))

            if result is not None:
                person_id = result["person_id"]
                dist = result["distance"]
                status = "NEW" if result["is_new"] else "MATCH"

                draw_label(
                    frame,
                    f"SIGMA ID {person_id} | {status} | d={dist:.2f}",
                    x1,
                    max(40, y1 - 8),
                    (0, 255, 0)
                )
            else:
                draw_label(
                    frame,
                    "Collecting gait sequence...",
                    x1,
                    max(40, y1 - 8),
                    (255, 255, 0)
                )

        if face_result.detections:
            for detection in face_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                draw_label(frame, "FACE", x, max(20, y - 10), (255, 0, 0))

        draw_label(
            frame,
            f"{cam_name} | Tracks: {len(tracks)} | DB IDs: {len(database.signatures)}",
            20,
            30,
            (0, 255, 0)
        )

        cv2.imshow(cam_name, frame)

    if all_failed:
        print("[ERROR] No frames available from any camera.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
database.save()
cap_laptop.release()
cap_phone.release()
cv2.destroyAllWindows()