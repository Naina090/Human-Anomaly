"""Optional YOLO pose inference plus an explainable, unvalidated fall heuristic."""
import os
import importlib.util


class DetectorUnavailable(RuntimeError):
    pass


def detector_status(model_path):
    """Describe actual optional detector setup without loading large model weights."""
    if not model_path:
        return {"configured": False, "ready": False, "message": "Validation only · no model configured"}
    has_package = importlib.util.find_spec("ultralytics") is not None
    valid_alias = model_path == "yolo26n-pose.pt"
    weights_exist = valid_alias or os.path.isfile(model_path)
    if not has_package:
        message = "AI package missing · install requirements-ai.txt"
    elif not weights_exist:
        message = "Pose weights not found · check POSE_MODEL_PATH"
    elif valid_alias:
        message = "Pose model configured · weights download on first run"
    else:
        message = "Pose model configured · ready for analysis"
    return {"configured": True, "ready": has_package and weights_exist, "message": message}


class YoloPoseFallDetector:
    """COCO-17 pose frontend. It detects a transition, never a single low posture."""

    METHOD = "YOLO pose + temporal fall-transition heuristic (research prototype)"

    def __init__(self, model_path):
        can_autodownload = model_path == "yolo26n-pose.pt"
        if not model_path or (not can_autodownload and not os.path.isfile(model_path)):
            raise DetectorUnavailable("Pose model weights are not available at the configured path.")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as exc:
            raise DetectorUnavailable("Pose model could not be loaded. Install the optional AI dependencies and check the weights.") from exc
        self.previous = None
        self.track_center = None
        self.upright_seen = False
        self.horizontal_since = None
        self.drop_seen_at = None
        self.drop_fraction = None
        self.emitted = False

    def inspect(self, frame, timestamp):
        """Return overlay and an event only after upright->drop->persistent posture evidence."""
        results = self.model.predict(frame, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            self.previous = None
            self.track_center = None
            self.upright_seen = False
            self.horizontal_since = None
            self.drop_seen_at = None
            self.drop_fraction = None
            return results[0].plot() if results else frame, None
        result = results[0]
        pts = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        # Keep association with the nearest previously observed body center to reduce
        # accidental transitions when several people are visible.
        centers = pts[:, [5, 6, 11, 12], :].mean(axis=1)
        if self.track_center is not None:
            idx = int((((centers-self.track_center) ** 2).sum(axis=1)).argmin())
        else:
            idx = int(confs.mean(axis=1).argmax()) if confs is not None else 0
        k = pts[idx]
        if confs is not None and float(confs[idx, [5, 6, 11, 12]].mean()) < .25:
            self.previous = None
            self.track_center = None
            self.upright_seen = False
            self.horizontal_since = None
            self.drop_seen_at = None
            self.drop_fraction = None
            return result.plot(), None
        # COCO keypoints: shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16.
        valid = lambda ids: all(float(k[i][1]) > 0 for i in ids)
        if not (valid((5, 6, 11, 12))):
            self.previous = None
            return result.plot(), None
        shoulder = k[[5, 6]].mean(axis=0)
        hip = k[[11, 12]].mean(axis=0)
        torso = hip - shoulder
        torso_ratio = abs(float(torso[0])) / max(abs(float(torso[1])), 1.0)
        center_y = float(k[[5, 6, 11, 12]].mean(axis=0)[1])
        self.track_center = k[[5, 6, 11, 12]].mean(axis=0)
        frame_height = frame.shape[0]
        upright = torso_ratio < 0.65
        horizontal = torso_ratio > 1.15
        prior = self.previous
        if upright and center_y < frame_height * 0.72:
            self.upright_seen = True
        dropped = bool(prior and timestamp - prior[0] <= 1.5 and
                       center_y - prior[1] > frame_height * 0.12)
        if dropped and self.upright_seen:
            self.drop_seen_at = timestamp
            self.drop_fraction = float((center_y - prior[1]) / frame_height)
        if horizontal:
            if self.horizontal_since is None:
                self.horizontal_since = timestamp
        else:
            self.horizontal_since = None
        persistent = self.horizontal_since is not None and timestamp - self.horizontal_since >= 0.6
        event = None
        if self.upright_seen and self.drop_seen_at is not None and timestamp-self.drop_seen_at <= 1.5 and persistent and not self.emitted:
            drop_score = min(1.0, max(0.0, (self.drop_fraction or 0) / .3))
            orient_score = min(1.0, torso_ratio / 2.0)
            persistence_score = min(1.0, (timestamp - self.horizontal_since) / 1.2)
            score = .5 * drop_score + .3 * orient_score + .2 * persistence_score
            event = {"event_type": "potential_fall", "timestamp_seconds": float(timestamp),
                     "confidence": None, "anomaly_score": round(float(score), 4),
                     "detection_method": self.METHOD,
                     "metadata": {"vertical_drop_fraction": round(float(self.drop_fraction or 0), 4),
                                  "torso_horizontal_ratio": round(torso_ratio, 4),
                                  "horizontal_persistence_seconds": round(timestamp-self.horizontal_since, 3),
                                  "score_components": "0.5*drop + 0.3*orientation + 0.2*persistence"}}
            self.emitted = True
        self.previous = (timestamp, center_y)
        return result.plot(), event
