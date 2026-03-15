import json
import os
import numpy as np


class GaitDatabase:
    def __init__(self, db_path="data/gait/gait_db.json", threshold=8.0):
        self.db_path = db_path
        self.threshold = threshold
        self.signatures = {}
        self.next_id = 1
        self._load()

    def _load(self):
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.threshold = data.get("threshold", self.threshold)
            raw_signatures = data.get("signatures", {})
            self.signatures = {
                int(pid): np.array(sig, dtype=np.float32)
                for pid, sig in raw_signatures.items()
            }

            if self.signatures:
                self.next_id = max(self.signatures.keys()) + 1
        except Exception as e:
            print(f"[WARN] Failed to load gait database: {e}")
            self.signatures = {}
            self.next_id = 1

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "threshold": self.threshold,
            "signatures": {
                str(pid): sig.tolist()
                for pid, sig in self.signatures.items()
            }
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def match(self, signature):
        signature = np.asarray(signature, dtype=np.float32)

        best_id = None
        best_dist = float("inf")

        for pid, known_sig in self.signatures.items():
            dist = np.linalg.norm(signature - known_sig)
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is not None and best_dist < self.threshold:
            return best_id, float(best_dist), False

        pid = self.next_id
        self.signatures[pid] = signature
        self.next_id += 1
        self.save()
        return pid, float(best_dist), True

    def update_signature(self, pid, signature, alpha=0.2):
        signature = np.asarray(signature, dtype=np.float32)

        if pid not in self.signatures:
            self.signatures[pid] = signature
        else:
            self.signatures[pid] = (
                (1 - alpha) * self.signatures[pid] + alpha * signature
            )

        self.save()