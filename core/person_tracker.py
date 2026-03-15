import numpy as np


class SimplePersonTracker:
    def __init__(self, max_distance=100, max_missing=15):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections: list of dicts
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": np.array([cx, cy]),
                    "landmarks": landmarks
                }
            ]
        """

        assigned_tracks = set()
        assigned_detections = set()

        track_ids = list(self.tracks.keys())

        for track_id in track_ids:
            track = self.tracks[track_id]
            best_det_idx = None
            best_dist = float("inf")

            for i, det in enumerate(detections):
                if i in assigned_detections:
                    continue

                dist = np.linalg.norm(det["center"] - track["center"])
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = i

            if best_det_idx is not None and best_dist <= self.max_distance:
                det = detections[best_det_idx]
                self.tracks[track_id]["bbox"] = det["bbox"]
                self.tracks[track_id]["center"] = det["center"]
                self.tracks[track_id]["landmarks"] = det["landmarks"]
                self.tracks[track_id]["missing"] = 0

                assigned_tracks.add(track_id)
                assigned_detections.add(best_det_idx)
            else:
                self.tracks[track_id]["missing"] += 1

        for i, det in enumerate(detections):
            if i in assigned_detections:
                continue

            self.tracks[self.next_track_id] = {
                "bbox": det["bbox"],
                "center": det["center"],
                "landmarks": det["landmarks"],
                "missing": 0
            }
            self.next_track_id += 1

        dead_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missing"] > self.max_missing
        ]
        for track_id in dead_tracks:
            del self.tracks[track_id]

        return self.tracks