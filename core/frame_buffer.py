from collections import deque
import numpy as np


class FrameBuffer:
    def __init__(self, max_frames=30, signature_window=5):
        self.max_frames = max_frames
        self.buffer = deque(maxlen=max_frames)
        self.landmarks = deque(maxlen=max_frames)
        self.signature_buffer = deque(maxlen=signature_window)

    def add(self, gait_vector, landmarks):
        self.buffer.append(gait_vector)
        self.landmarks.append(landmarks)

    def is_full(self):
        return len(self.buffer) == self.max_frames

    def clear(self):
        self.buffer.clear()
        self.landmarks.clear()
        self.signature_buffer.clear()

    def get_sequence(self):
        return np.array(self.buffer, dtype=np.float32)

    def get_landmarks(self):
        return list(self.landmarks)

    def add_signature(self, signature):
        self.signature_buffer.append(np.asarray(signature, dtype=np.float32))

    def get_stable_signature(self, min_samples=3):
        """Returns the mean signature once at least min_samples have been accumulated."""
        if len(self.signature_buffer) < min_samples:
            return None
        return np.mean(np.array(self.signature_buffer), axis=0)