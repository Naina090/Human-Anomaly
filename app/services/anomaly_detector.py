"""Explainable posture/motion heuristic. It does not classify falls or violence."""


def classify_frame(gray, timestamp, previous=None):
    """Return a motion-change score from sampled grayscale frames, or None.

    This is a scene-motion change signal only. It is not person detection, pose
    estimation, or a fall prediction and is intentionally not saved as an event.
    """
    if previous is None:
        return None
    import cv2
    import numpy as np
    a = cv2.resize(previous, (96, 96)).astype(np.float32)
    b = cv2.resize(gray, (96, 96)).astype(np.float32)
    score = float(np.mean(np.abs(a - b)) / 255.0)
    return {"timestamp": timestamp, "score": round(score, 4), "method": "scene_motion_change"}
