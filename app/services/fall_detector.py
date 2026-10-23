"""Optional local Ultralytics pose inference and explainable fall heuristic."""
import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version


class DetectorUnavailable(RuntimeError):
    pass


COCO_JOINTS = {
    5: "left_shoulder", 6: "right_shoulder",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}
METHOD = "YOLO pose + temporal fall-transition heuristic (research prototype)"


def detector_status(model_path):
    """Report availability without loading weights or downloading anything."""
    model_path = os.path.expanduser(str(model_path or "").strip())
    if not model_path:
        return {"configured": False, "ready": False, "status": "not_configured",
                "method": METHOD, "message": "Validation only · no pose model configured"}
    if importlib.util.find_spec("ultralytics") is None:
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Pose package unavailable · install requirements-ai.txt"}
    if not os.path.isfile(model_path):
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Local pose weights not found · set POSE_MODEL_PATH to an existing file"}
    return {"configured": True, "ready": True, "status": "ready", "method": METHOD,
            "message": "Local pose model ready for analysis"}


class YoloPoseFallDetector:
    """COCO pose frontend; a single low posture can never emit a fall candidate."""

    METHOD = METHOD

    def __init__(self, model_path, settings=None):
        model_path = os.path.abspath(os.path.expanduser(str(model_path or "")))
        if not model_path or not os.path.isfile(model_path):
            raise DetectorUnavailable("Pose model weights were not found at the configured local path.")
        if importlib.util.find_spec("ultralytics") is None:
            raise DetectorUnavailable("Install the optional requirements-ai.txt dependencies to use pose analysis.")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as exc:
            raise DetectorUnavailable("Pose model could not be loaded. Check the local weights and AI dependencies.") from exc
        try:
            self.model_version = version("ultralytics")
        except PackageNotFoundError:
            self.model_version = None
        self.model_name = os.path.basename(model_path)
        settings = settings or {}
        self.transition_window_seconds = max(.2, float(settings.get("transition_window_seconds", 1.5)))
        self.minimum_vertical_drop = max(.01, min(.8, float(settings.get("minimum_vertical_drop", .12))))
        self.minimum_horizontal_persistence = max(.2, float(settings.get("minimum_horizontal_persistence", .6)))
        self.candidate_score_threshold = max(0, min(1, float(settings.get("candidate_score_threshold", .55))))
        self.previous = None
        self.track_center = None
        self.upright_seen = False
        self.upright_timestamp = None
        self.horizontal_since = None
        self.drop_seen_at = None
        self.drop_fraction = None
        self.drop_start_at = None
        self.emitted = False

    def inspect(self, frame, timestamp):
        """Compatibility interface for earlier callers: return overlay and candidate."""
        overlay, event, _evidence = self.inspect_with_evidence(frame, timestamp)
        return overlay, event

    def inspect_with_evidence(self, frame, timestamp):
        """Return annotated frame, optional candidate, and compact sampled pose evidence."""
        results = self.model.predict(frame, verbose=False)
        if not results:
            self._reset_transition()
            return frame, None, self._empty_evidence(timestamp)
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        xy = getattr(keypoints, "xy", None) if keypoints is not None else None
        if xy is None:
            self._reset_transition()
            return result.plot(), None, self._empty_evidence(timestamp)
        pts = xy.cpu().numpy() if hasattr(xy, "cpu") else xy.numpy() if hasattr(xy, "numpy") else xy
        conf_tensor = getattr(keypoints, "conf", None)
        confs = conf_tensor.cpu().numpy() if conf_tensor is not None and hasattr(conf_tensor, "cpu") else (
            conf_tensor.numpy() if conf_tensor is not None and hasattr(conf_tensor, "numpy") else conf_tensor)
        if len(pts) == 0:
            self._reset_transition()
            return result.plot(), None, self._empty_evidence(timestamp)
        evidence_people = []
        for person_idx, person_points in enumerate(pts):
            confidence_row = confs[person_idx] if confs is not None and person_idx < len(confs) else None
            joints = {}
            for point_idx, joint_name in COCO_JOINTS.items():
                point = person_points[point_idx]
                confidence = float(confidence_row[point_idx]) if confidence_row is not None else None
                joints[joint_name] = {
                    "x": round(float(point[0]), 2), "y": round(float(point[1]), 2),
                    "confidence": round(confidence, 5) if confidence is not None else None,
                }
            evidence_people.append({"keypoints": joints})

        # Continue the fall heuristic with the pose having the strongest shoulder/hip evidence.
        centers = pts[:, [5, 6, 11, 12], :].mean(axis=1)
        if self.track_center is not None:
            idx = int((((centers - self.track_center) ** 2).sum(axis=1)).argmin())
        else:
            idx = int(confs.mean(axis=1).argmax()) if confs is not None else 0
        k = pts[idx]
        confidence_mean = float(confs[idx, [5, 6, 11, 12]].mean()) if confs is not None else None
        evidence = {"timestamp_seconds": float(timestamp), "person_count": int(len(pts)),
                    "model_name": getattr(self, "model_name", None),
                    "model_version": getattr(self, "model_version", None),
                    "keypoint_names": list(COCO_JOINTS.values()), "people": evidence_people,
                    "selected_person_index": idx,
                    "selected_torso_confidence": round(confidence_mean, 5) if confidence_mean is not None else None}
        if confidence_mean is not None and confidence_mean < .25:
            self._reset_transition()
            evidence["transition_status"] = "inconclusive_low_confidence"
            return result.plot(), None, evidence
        if not all(float(k[i][1]) >= 0 for i in (5, 6, 11, 12)):
            self._reset_transition()
            evidence["transition_status"] = "inconclusive_missing_torso"
            return result.plot(), None, evidence

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
            self.upright_timestamp = float(timestamp)
        dropped = bool(prior and 0 <= timestamp - prior[0] <= self.transition_window_seconds and
                       center_y - prior[1] > frame_height * self.minimum_vertical_drop)
        if dropped and self.upright_seen:
            self.drop_seen_at = float(timestamp)
            self.drop_start_at = float(prior[0])
            self.drop_fraction = float((center_y - prior[1]) / frame_height)
        if horizontal:
            if self.horizontal_since is None:
                self.horizontal_since = float(timestamp)
        else:
            self.horizontal_since = None
        persistent = (self.horizontal_since is not None and
                      timestamp - self.horizontal_since >= self.minimum_horizontal_persistence)
        event = None
        if (self.upright_seen and self.drop_seen_at is not None and
                timestamp - self.drop_seen_at <= self.transition_window_seconds and persistent and not self.emitted):
            drop_score = min(1.0, max(0.0, (self.drop_fraction or 0) / .3))
            orient_score = min(1.0, torso_ratio / 2.0)
            persistence_score = min(1.0, (timestamp - self.horizontal_since) / 1.2)
            score = .5 * drop_score + .3 * orient_score + .2 * persistence_score
            if score < self.candidate_score_threshold:
                self.previous = (float(timestamp), center_y)
                evidence["transition_status"] = "candidate_below_score_threshold"
                evidence["anomaly_score"] = round(float(score), 4)
                return result.plot(), None, evidence
            event = {
                "event_type": "potential_fall", "label": "Potential fall candidate",
                "timestamp_seconds": float(timestamp),
                "start_timestamp_seconds": float(self.drop_start_at if self.drop_start_at is not None else self.drop_seen_at),
                "end_timestamp_seconds": float(timestamp),
                "confidence": None, "anomaly_score": round(float(score), 4),
                "detection_method": self.METHOD,
                "details": {
                    "previously_upright": True,
                    "upright_timestamp_seconds": self.upright_timestamp,
                    "rapid_drop_timestamp_seconds": self.drop_seen_at,
                    "vertical_drop_fraction": round(float(self.drop_fraction or 0), 4),
                    "torso_horizontal_ratio": round(torso_ratio, 4),
                    "horizontal_persistence_seconds": round(timestamp - self.horizontal_since, 3),
                    "score_components": "0.5*drop + 0.3*orientation + 0.2*persistence",
                    "model_name": getattr(self, "model_name", None),
                    "model_version": getattr(self, "model_version", None),
                    "candidate_score_threshold": self.candidate_score_threshold,
                    "limitations": "Unvalidated heuristic; sitting, bending, intentional lying, occlusion, and tracking switches can affect results.",
                },
            }
            self.emitted = True
        self.previous = (float(timestamp), center_y)
        evidence["transition_status"] = "candidate_emitted" if event else (
            "upright_seen" if self.upright_seen else "observed_no_candidate")
        evidence["torso_horizontal_ratio"] = round(torso_ratio, 4)
        return result.plot(), event, evidence

    @staticmethod
    def _empty_evidence(timestamp):
        return {"timestamp_seconds": float(timestamp), "person_count": 0, "people": [],
                "transition_status": "no_person_observed"}

    def _reset_transition(self):
        self.previous = None
        self.track_center = None
        self.upright_seen = False
        self.upright_timestamp = None
        self.horizontal_since = None
        self.drop_seen_at = None
        self.drop_fraction = None
        self.drop_start_at = None
