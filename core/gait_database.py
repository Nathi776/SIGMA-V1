import json
import os
import time
import numpy as np


class GaitDatabase:
    def __init__(self, db_path="data/gait/gait_db.json", threshold=0.15):
        self.db_path = db_path
        self.threshold = threshold
        self.signatures = {}
        self.profiles = {}  # pid -> {"name": str, "created_at": str, "observations": int}
        self.next_id = 1
        self._load()

    def _normalize(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

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
                int(pid): self._normalize(np.array(sig, dtype=np.float32))
                for pid, sig in raw_signatures.items()
            }

            raw_profiles = data.get("profiles", {})
            self.profiles = {
                int(pid): prof
                for pid, prof in raw_profiles.items()
            }

            # Ensure all signatures have corresponding profile entries
            for pid in self.signatures:
                if pid not in self.profiles:
                    self.profiles[pid] = {
                        "name": f"Person {pid}",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "observations": 1
                    }

            # Filter out legacy signatures whose dimensionality does not match the 419-D schema
            legacy_pids = [pid for pid, s in self.signatures.items() if len(s) != 419]
            if legacy_pids:
                print(f"[INFO] Purging {len(legacy_pids)} legacy signatures with outdated feature dimensions.")
                for pid in legacy_pids:
                    del self.signatures[pid]
                    if pid in self.profiles:
                        del self.profiles[pid]
                self.save()

            if self.signatures:
                self.next_id = max(self.signatures.keys()) + 1
        except Exception as e:
            print(f"[WARN] Failed to load gait database: {e}")
            self.signatures = {}
            self.profiles = {}
            self.next_id = 1

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "threshold": self.threshold,
            "signatures": {
                str(pid): sig.tolist()
                for pid, sig in self.signatures.items()
            },
            "profiles": {
                str(pid): prof
                for pid, prof in self.profiles.items()
            }
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_name(self, pid):
        """Returns human-readable name for a person ID."""
        if pid in self.profiles:
            return self.profiles[pid].get("name", f"Person {pid}")
        return f"Person {pid}"

    def set_name(self, pid, name):
        """Assigns a custom name to an enrolled identity."""
        if pid in self.profiles:
            self.profiles[pid]["name"] = str(name)
            self.save()

    def compute_distance(self, sig1, sig2):
        """Computes cosine distance between two signatures: 0.0 (identical) to 2.0 (opposite)."""
        u = self._normalize(sig1)
        v = self._normalize(sig2)
        if u.shape != v.shape:
            # Different dimensionality from an older schema/version
            return float("inf")
        dot_product = float(np.dot(u, v))
        # Clamp to [-1.0, 1.0] for numerical stability
        dot_product = max(-1.0, min(1.0, dot_product))
        return 1.0 - dot_product

    def get_distance(self, pid, signature):
        """Returns distance between a known person_id and a candidate signature."""
        if pid not in self.signatures:
            return float("inf")
        return self.compute_distance(self.signatures[pid], signature)

    def match(self, signature, auto_enroll=True):
        """
        Matches a signature against known identities.
        Returns: (person_id, distance, is_new)
        """
        signature = self._normalize(signature)

        best_id = None
        best_dist = float("inf")

        for pid, known_sig in self.signatures.items():
            dist = self.compute_distance(signature, known_sig)
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is not None and best_dist < self.threshold:
            return best_id, float(best_dist), False

        if auto_enroll:
            pid = self.next_id
            self.signatures[pid] = signature
            self.profiles[pid] = {
                "name": f"Person {pid}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "observations": 1
            }
            self.next_id += 1
            self.save()
            return pid, float(best_dist), True

        return None, float(best_dist), False

    def update_signature(self, pid, signature, alpha=0.05):
        """Smoothly updates a known identity's signature using exponential moving average."""
        signature = self._normalize(signature)

        if pid not in self.signatures:
            self.signatures[pid] = signature
            self.profiles[pid] = {
                "name": f"Person {pid}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "observations": 1
            }
        else:
            updated = (1 - alpha) * self.signatures[pid] + alpha * signature
            self.signatures[pid] = self._normalize(updated)
            if pid in self.profiles:
                self.profiles[pid]["observations"] = self.profiles[pid].get("observations", 1) + 1

        self.save()

    def clear(self):
        """Resets the database."""
        self.signatures = {}
        self.next_id = 1
        self.save()