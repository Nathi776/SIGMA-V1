import numpy as np
import time
import os
import argparse
from core.gait_database import GaitDatabase
from core.gait_utils import normalize_signature


def test_signature_math():
    print("=" * 60)
    print("RUNNING GAIT MATHEMATICAL & MATCHING VALIDATION")
    print("=" * 60)

    test_db_path = "data/gait/test_temp_db.json"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = GaitDatabase(db_path=test_db_path, threshold=0.25)

    # 1. Generate synthetic baseline signature for Person A (419 dims)
    np.random.seed(42)
    sig_person_a = np.random.randn(419).astype(np.float32)

    # First match should auto-enroll Person 1
    pid, dist, is_new = db.match(sig_person_a, auto_enroll=True)
    print(f"Test 1 - Initial enrollment: Assigned ID={pid}, dist={dist:.4f}, is_new={is_new}")
    assert pid == 1, "Expected Person ID 1"
    assert is_new is True, "Expected is_new=True for initial enrollment"

    # 2. Perturb Person A's signature slightly (simulating slight gait noise across steps)
    noise_a = np.random.normal(0, 0.05, 419).astype(np.float32)
    sig_person_a_noisy = sig_person_a + noise_a

    pid_noisy, dist_noisy, is_new_noisy = db.match(sig_person_a_noisy, auto_enroll=True)
    print(f"Test 2 - Same person (noisy step): Matched ID={pid_noisy}, dist={dist_noisy:.4f}, is_new={is_new_noisy}")
    assert pid_noisy == 1, f"Expected matched ID 1, got {pid_noisy}"
    assert is_new_noisy is False, "Expected is_new=False for the same person"
    assert dist_noisy < db.threshold, f"Distance {dist_noisy:.4f} should be below threshold {db.threshold}"

    # 3. Generate a distinct signature for Person B
    sig_person_b = np.random.randn(419).astype(np.float32) + 2.0

    pid_b, dist_b, is_new_b = db.match(sig_person_b, auto_enroll=True)
    print(f"Test 3 - Distinct person B: Assigned ID={pid_b}, dist={dist_b:.4f}, is_new={is_new_b}")
    assert pid_b == 2, f"Expected new Person ID 2, got {pid_b}"
    assert is_new_b is True, "Expected is_new=True for distinct person"
    assert dist_b >= db.threshold, f"Distance {dist_b:.4f} should exceed threshold"

    # 4. Re-test Person A after Person B was enrolled
    pid_a_retest, dist_a_retest, is_new_a_retest = db.match(sig_person_a, auto_enroll=True)
    print(f"Test 4 - Retest Person A: Matched ID={pid_a_retest}, dist={dist_a_retest:.4f}, is_new={is_new_a_retest}")
    assert pid_a_retest == 1, f"Expected Person 1, got {pid_a_retest}"
    assert is_new_a_retest is False, "Expected is_new=False"

    # Cleanup temp db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    print("\n[SUCCESS] All mathematical and database matching tests passed!")
    print("Runaway ID problem is resolved: same person remains stable, different persons get distinct IDs.")


if __name__ == "__main__":
    test_signature_math()
