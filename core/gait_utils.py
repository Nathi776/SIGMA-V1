import numpy as np


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _point_xyz(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _point_xy(lm):
    return np.array([lm.x, lm.y], dtype=np.float32)


def compute_angle(a, b, c):
    a = _point_xy(a)
    b = _point_xy(b)
    c = _point_xy(c)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def normalize_pose(landmarks):
    left_hip = _point_xyz(landmarks[LEFT_HIP])
    right_hip = _point_xyz(landmarks[RIGHT_HIP])
    hip_center = (left_hip + right_hip) / 2.0

    left_shoulder = _point_xyz(landmarks[LEFT_SHOULDER])
    right_shoulder = _point_xyz(landmarks[RIGHT_SHOULDER])
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    left_ankle = _point_xyz(landmarks[LEFT_ANKLE])
    right_ankle = _point_xyz(landmarks[RIGHT_ANKLE])
    ankle_center = (left_ankle + right_ankle) / 2.0

    torso_scale = np.linalg.norm(shoulder_center - hip_center)
    leg_scale = np.linalg.norm(ankle_center - hip_center)
    scale = max(torso_scale + leg_scale, 1e-6)

    vector = []
    for lm in landmarks:
        p = _point_xyz(lm)
        p = (p - hip_center) / scale
        vector.extend(p.tolist())

    return np.array(vector, dtype=np.float32)


def extract_gait_signature(sequence):
    if len(sequence) < 2:
        return None

    velocity = np.diff(sequence, axis=0)
    acceleration = np.diff(velocity, axis=0) if len(velocity) >= 2 else np.zeros_like(velocity)

    mean_pose = np.mean(sequence, axis=0)
    mean_velocity = np.mean(velocity, axis=0)
    std_velocity = np.std(velocity, axis=0)
    mean_acceleration = np.mean(acceleration, axis=0) if len(acceleration) > 0 else np.zeros(sequence.shape[1])

    return np.concatenate([
        mean_pose,
        mean_velocity,
        std_velocity,
        mean_acceleration
    ]).astype(np.float32)


def extract_biomechanical_features(landmark_sequence):
    if not landmark_sequence:
        return np.zeros(10, dtype=np.float32)

    left_knee_angles = []
    right_knee_angles = []
    left_hip_y = []
    right_hip_y = []
    ankle_distance = []

    for landmarks in landmark_sequence:
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        left_knee = landmarks[LEFT_KNEE]
        right_knee = landmarks[RIGHT_KNEE]
        left_ankle = landmarks[LEFT_ANKLE]
        right_ankle = landmarks[RIGHT_ANKLE]

        left_knee_angles.append(compute_angle(left_hip, left_knee, left_ankle))
        right_knee_angles.append(compute_angle(right_hip, right_knee, right_ankle))

        left_hip_y.append(left_hip.y)
        right_hip_y.append(right_hip.y)

        ankle_distance.append(abs(left_ankle.x - right_ankle.x))

    left_knee_angles = np.array(left_knee_angles, dtype=np.float32)
    right_knee_angles = np.array(right_knee_angles, dtype=np.float32)
    hip_y = (np.array(left_hip_y, dtype=np.float32) + np.array(right_hip_y, dtype=np.float32)) / 2.0
    ankle_distance = np.array(ankle_distance, dtype=np.float32)

    centered_hip = hip_y - np.mean(hip_y)
    zero_crossings = np.where(np.diff(np.sign(centered_hip)) != 0)[0]
    step_frequency = len(zero_crossings) / max(len(hip_y), 1)

    features = np.array([
        left_knee_angles.mean(),
        left_knee_angles.std(),
        right_knee_angles.mean(),
        right_knee_angles.std(),
        ((left_knee_angles + right_knee_angles) / 2.0).mean(),
        ((left_knee_angles + right_knee_angles) / 2.0).std(),
        step_frequency,
        hip_y.std(),
        ankle_distance.mean(),
        ankle_distance.std(),
    ], dtype=np.float32)

    return features


def landmarks_to_bbox(landmarks, frame_w, frame_h, padding=20):
    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)
    x2 = min(max(xs) + padding, frame_w - 1)
    y2 = min(max(ys) + padding, frame_h - 1)

    return (x1, y1, x2, y2)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)