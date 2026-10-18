import time
import cv2


class VideoError(Exception):
    pass


def analyze_video(path, sample_seconds, max_duration, detector=None, progress=None, output_path=None):
    start = time.monotonic()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoError("OpenCV could not read this video. Try an MP4 encoded with H.264.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0:
        capture.release()
        raise VideoError("The file has no readable video frames.")
    if duration > max_duration:
        capture.release()
        raise VideoError(f"Video is {duration:.0f}s; maximum allowed is {max_duration}s.")
    # Read at a bounded interval without retaining a complete video in memory.
    interval = max(1, int(fps * sample_seconds))
    read = 0
    events = []
    frame_index = 0
    writer = None
    annotated = False
    if detector is not None and output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None
    try:
        while frame_index < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                read += 1
                if detector is not None:
                    frame, event = detector.inspect(frame, frame_index / fps)
                    annotated = writer is not None
                    if event:
                        events.append(event)
                if progress:
                    progress(min(95, int(95 * frame_index / max(frame_count, 1))))
            if writer is not None:
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if read == 0:
        raise VideoError("No readable frames were found in this file.")
    return {"duration": duration, "fps": fps, "width": width, "height": height,
            "processing_seconds": time.monotonic() - start, "sampled_frames": read,
            "events": events, "method": detector.METHOD if detector else "validation_only",
            "analysis_mode": "pose_heuristic" if detector else "validation_only",
            "annotated_output": bool(annotated)}
