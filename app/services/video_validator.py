"""Fast, model-independent checks for uploaded video files."""
import math

import cv2


class VideoValidationError(ValueError):
    """Raised when OpenCV cannot decode a usable frame from an upload."""


def validate_video_file(path):
    """Open the file, decode one frame, and return metadata reported by OpenCV.

    A filename extension is not treated as proof of valid video content. The
    capture is always released, including when a codec or read operation fails.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(
            "OpenCV could not open this video. Try an MP4 encoded with H.264 or a codec supported on this computer."
        )

    try:
        ok, frame = capture.read()
        if not ok or frame is None or not getattr(frame, "size", 0):
            raise VideoValidationError(
                "The file opened, but OpenCV could not decode a video frame. The video may be corrupted or use an unsupported codec."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
        if not math.isfinite(fps) or fps <= 0:
            fps = None
        if frame_count <= 0:
            frame_count = None
        duration = frame_count / fps if frame_count and fps else None
        return {
            "duration": duration,
            "frame_rate": fps,
            "frame_count": frame_count,
            "width": width or None,
            "height": height or None,
        }
    except cv2.error as exc:
        raise VideoValidationError(
            "OpenCV encountered a decoding error. Try re-exporting the video with a supported codec."
        ) from exc
    finally:
        capture.release()
