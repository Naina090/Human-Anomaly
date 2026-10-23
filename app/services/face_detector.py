"""OpenCV frontal-face detection using its bundled Haar cascade.

This detector locates visible frontal faces. It does not identify people.
"""
import os

import cv2


METHOD = "OpenCV Haar cascade (haarcascade_frontalface_default)"


class FaceDetectorUnavailable(RuntimeError):
    """Raised when the configured cascade cannot be loaded."""


def cascade_path(configured_path=""):
    """Use an explicit cascade path, or the file bundled with opencv-python."""
    if configured_path:
        return os.path.abspath(os.path.expanduser(configured_path))
    bundled_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if bundled_dir:
        return os.path.join(bundled_dir, "haarcascade_frontalface_default.xml")
    return ""


def detector_status(configured_path=""):
    path = cascade_path(configured_path)
    if not path or not os.path.isfile(path):
        return {"ready": False, "method": METHOD,
                "message": "Face detector unavailable · OpenCV frontal-face cascade was not found"}
    try:
        if cv2.CascadeClassifier(path).empty():
            raise ValueError("empty cascade")
    except Exception:
        return {"ready": False, "method": METHOD,
                "message": "Face detector unavailable · configured OpenCV cascade could not be loaded"}
    return {"ready": True, "method": METHOD,
            "message": "OpenCV frontal-face detector ready · no identity recognition"}


class OpenCVFaceDetector:
    """Detect multiple frontal faces and return boxes without confidence scores."""

    METHOD = METHOD

    def __init__(self, configured_path=""):
        path = cascade_path(configured_path)
        if not path or not os.path.isfile(path):
            raise FaceDetectorUnavailable(
                "The OpenCV frontal-face cascade is unavailable. Install the standard opencv-python package or set FACE_CASCADE_PATH to a valid cascade XML file."
            )
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise FaceDetectorUnavailable(
                "The configured OpenCV face cascade could not be loaded. Check FACE_CASCADE_PATH."
            )

    def detect(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )
        return [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                for x, y, w, h in boxes]

    @staticmethod
    def annotate(frame, boxes):
        annotated = frame.copy()
        for box in boxes:
            x, y = box["x"], box["y"]
            cv2.rectangle(annotated, (x, y), (x + box["width"], y + box["height"]),
                          (40, 210, 235), 2)
        return annotated
