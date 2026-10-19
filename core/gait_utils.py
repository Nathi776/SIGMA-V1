import numpy as np


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
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
    """
    Extracts 22 normalized biomechanical features covering:
    - Knee kinematics (mean, std, range of motion, symmetry)
    - Hip kinematics (flexion/extension mean, std, range of motion, symmetry)
    - Arm swing dynamics (wrist-to-hip amplitude, swing symmetry)
    - Posture & torso forward lean
    - Cadence & stride dynamics
    All features are normalized into [0.0, 1.0].
    """
    if not landmark_sequence:
        return np.zeros(22, dtype=np.float32)

    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    torso_tilt_angles = []
    left_arm_swings = []
    right_arm_swings = []
    ankle_sep_x = []
    hip_y = []

    for landmarks in landmark_sequence:
        l_sh = landmarks[LEFT_SHOULDER]
        r_sh = landmarks[RIGHT_SHOULDER]
        l_hip = landmarks[LEFT_HIP]
        r_hip = landmarks[RIGHT_HIP]
        l_knee = landmarks[LEFT_KNEE]
        r_knee = landmarks[RIGHT_KNEE]
        l_ank = landmarks[LEFT_ANKLE]
        r_ank = landmarks[RIGHT_ANKLE]
        l_wri = landmarks[LEFT_WRIST]
        r_wri = landmarks[RIGHT_WRIST]

        # Knee angles (degrees)
        left_knee_angles.append(compute_angle(l_hip, l_knee, l_ank))
        right_knee_angles.append(compute_angle(r_hip, r_knee, r_ank))

        # Hip angles (torso to thigh)
        left_hip_angles.append(compute_angle(l_sh, l_hip, l_knee))
        right_hip_angles.append(compute_angle(r_sh, r_hip, r_knee))

        # Torso posture (angle of spine vector relative to vertical)
        sh_center = (_point_xy(l_sh) + _point_xy(r_sh)) / 2.0
        h_center = (_point_xy(l_hip) + _point_xy(r_hip)) / 2.0
        spine = sh_center - h_center
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        dot_v = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-6)
        torso_tilt = np.degrees(np.arccos(np.clip(dot_v, -1.0, 1.0)))
        torso_tilt_angles.append(torso_tilt)

        # Arm swing: 2D distance between wrist and hip (in normalized space)
        left_arm_swings.append(np.linalg.norm(_point_xy(l_wri) - _point_xy(l_hip)))
        right_arm_swings.append(np.linalg.norm(_point_xy(r_wri) - _point_xy(r_hip)))

        # Ankle stride separation
        ankle_sep_x.append(l_ank.x - r_ank.x)
        hip_y.append((l_hip.y + r_hip.y) / 2.0)

    lk = np.array(left_knee_angles, dtype=np.float32)
    rk = np.array(right_knee_angles, dtype=np.float32)
    lh = np.array(left_hip_angles, dtype=np.float32)
    rh = np.array(right_hip_angles, dtype=np.float32)
    tt = np.array(torso_tilt_angles, dtype=np.float32)
    las = np.array(left_arm_swings, dtype=np.float32)
    ras = np.array(right_arm_swings, dtype=np.float32)
    asx = np.array(ankle_sep_x, dtype=np.float32)
    hy = np.array(hip_y, dtype=np.float32)

    # Knee dynamics (normalized by 180.0)
    lk_mean = lk.mean() / 180.0
    lk_std = lk.std() / 180.0
    lk_rom = (lk.max() - lk.min()) / 180.0 if len(lk) > 0 else 0.0

    rk_mean = rk.mean() / 180.0
    rk_std = rk.std() / 180.0
    rk_rom = (rk.max() - rk.min()) / 180.0 if len(rk) > 0 else 0.0

    knee_symmetry = abs(lk_mean - rk_mean)

    # Hip dynamics (normalized by 180.0)
    lh_mean = lh.mean() / 180.0
    lh_std = lh.std() / 180.0
    lh_rom = (lh.max() - lh.min()) / 180.0 if len(lh) > 0 else 0.0

    rh_mean = rh.mean() / 180.0
    rh_std = rh.std() / 180.0
    rh_rom = (rh.max() - rh.min()) / 180.0 if len(rh) > 0 else 0.0

    hip_symmetry = abs(lh_mean - rh_mean)

    # Torso posture (normalized by 90.0)
    torso_mean = tt.mean() / 90.0
    torso_std = tt.std() / 90.0

    # Arm swing dynamics (normalized coords ~ 0 to 1)
    las_mean = las.mean()
    las_std = las.std()
    ras_mean = ras.mean()
    ras_std = ras.std()
    arm_swing_symmetry = abs(las_mean - ras_mean)

    # Stride & Cadence (zero-crossings of ankle oscillation with amplitude deadband)
    centered_asx = asx - np.mean(asx)
    oscillation_amp = np.ptp(asx)
    if oscillation_amp > 0.04:  # True walking motion threshold
        zero_crossings = np.where(np.diff(np.sign(centered_asx)) != 0)[0]
        step_frequency = len(zero_crossings) / max(len(asx), 1)
    else:
        step_frequency = 0.0  # Stationary person has 0 cadence

    features = np.array([
        lk_mean, lk_std, lk_rom,
        rk_mean, rk_std, rk_rom,
        knee_symmetry,
        lh_mean, lh_std, lh_rom,
        rh_mean, rh_std, rh_rom,
        hip_symmetry,
        torso_mean, torso_std,
        las_mean, las_std,
        ras_mean, ras_std,
        arm_swing_symmetry,
        step_frequency,
        abs(asx).mean(),  # Mean stride width
    ], dtype=np.float32)

    return features


def normalize_signature(signature):
    """L2-normalizes a gait signature vector for consistent cosine/euclidean distance calculation."""
    sig = np.asarray(signature, dtype=np.float32)
    norm = np.linalg.norm(sig)
    if norm < 1e-6:
        return sig
    return sig / norm


def landmarks_to_bbox(landmarks, frame_w, frame_h, padding=20):
    if not landmarks:
        return (0, 0, 0, 0)
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