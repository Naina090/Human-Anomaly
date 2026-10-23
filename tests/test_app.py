import io
import os
import sys
import time
import types
from datetime import datetime, timezone

import cv2
import mongomock
import numpy as np
import pytest
from bson import ObjectId
from pymongo.errors import OperationFailure

from app import create_app
from app.db import DatabaseUnavailable, _friendly_connection_error
from app.services.repository import (
    create_video, dashboard_stats, delete_video, get_video, list_videos,
    object_id, report_data, update_video,
)
from app.services.fall_detector import YoloPoseFallDetector
from app.services.storage import safe_video_path, safe_processed_path
from app.services.face_detector import OpenCVFaceDetector, FaceDetectorUnavailable, detector_status as face_detector_status
from app.services.object_detector import (ObjectDetectorUnavailable, YoloObjectTracker,
    detector_status as object_detector_status)
from app.services.video_processor import analyze_video
from app.services.video_validator import validate_video_file, VideoValidationError
from app.services.report_service import build_analysis_summary, video_report_csv


@pytest.fixture()
def app(tmp_path):
    client = mongomock.MongoClient()
    application = create_app({
        "TESTING": True, "SECRET_KEY": "test-secret", "MONGO_CLIENT": client,
        "MONGO_DB_NAME": "test_human_anomaly", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "uploads"), "PROCESSED_DIR": str(tmp_path / "processed"),
        "MAX_CONTENT_LENGTH": 2_000_000, "MAX_DURATION_SECONDS": 30,
        "SAMPLE_EVERY_SECONDS": 0.1,
    })
    yield application
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    client.close()


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client):
    client.get("/upload")
    with client.session_transaction() as session:
        return session["csrf_token"]


def video_doc(name="clip.mp4", status="processing"):
    return {
        "original_filename": name, "stored_filename": "internal.mp4", "file_path": "internal.mp4",
        "upload_date": datetime.now(timezone.utc), "processing_status": status,
        "duration": None, "frame_rate": None, "width": None, "height": None,
        "processing_duration": None, "error_message": None,
    }


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def wait_for_status(app, video_id, statuses=("complete", "failed")):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = app.extensions["mongo_db"].videos.find_one({"_id": ObjectId(video_id)})
        if row and row["processing_status"] in statuses:
            return row
        time.sleep(.02)
    raise AssertionError("Background video job did not finish in time")


def test_app_startup_and_pages_render(client):
    for url in ("/", "/dashboard", "/about", "/history", "/reports", "/upload"):
        assert client.get(url).status_code == 200


def test_empty_history_shows_real_empty_state(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"No matching videos" in response.data
    assert b"will appear here" in response.data


@pytest.mark.parametrize("status", ["processing", "failed", "complete"])
def test_results_page_handles_legacy_video_without_optional_fields(client, app, status):
    # This reproduces the prior 500: results() used to call strftime() on the
    # absent upload_date field in older MongoDB documents.
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "legacy-video.mp4",
        "processing_status": status,
    }).inserted_id

    response = client.get(f"/results/{video_id}")

    assert response.status_code == 200
    assert b"legacy-video.mp4" in response.data
    assert b"Uploaded Unknown" in response.data
    assert b"No face results are stored" in response.data
    assert app.extensions["mongo_db"].detection_events.count_documents({"video_id": video_id}) == 0


def test_results_page_returns_404_for_invalid_or_missing_video(client):
    assert client.get("/results/not-an-object-id").status_code == 404
    assert client.get("/results/507f1f77bcf86cd799439011").status_code == 404


def test_history_page_handles_legacy_video_without_upload_date(client, app):
    app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "legacy-history-video.avi",
        "stored_filename": "legacy-internal.avi",
        "processing_status": "complete",
    })

    response = client.get("/history")

    assert response.status_code == 200
    assert b"legacy-history-video.avi" in response.data
    assert b"Unknown" in response.data


def test_dashboard_shows_database_and_detector_readiness_without_fake_counts(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"MongoDB" in response.data and b"Connected" in response.data
    assert b"Pose analysis" in response.data and b"Validation only" in response.data
    assert b">0</strong>" in response.data
    assert b"No videos yet" in response.data and b"No events recorded" in response.data


def test_mongodb_connection_failure_is_friendly(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": "mongodb://127.0.0.1:1/",
                      "MONGO_TIMEOUT_MS": 20, "MONGO_DB_NAME": "connection_test",
                      "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    client = app.test_client()
    for path in ("/dashboard", "/history", "/reports"):
        response = client.get(path)
        assert response.status_code == 503
        assert b"MongoDB is unavailable" in response.data
        assert b"MONGODB CONNECTION" in response.data
        assert b"temporarily unavailable" in response.data
        assert b"internal error" not in response.data.lower()
    assert client.get("/").status_code == 200
    app.extensions["mongo_client"].close()


def test_analyze_video_route_logs_unexpected_upload_failure_without_exposing_traceback(
        client, app, tmp_path, monkeypatch, caplog):
    import app.routes.videos as video_routes

    # Exercise the same /upload page and multipart POST used by Analyze/New
    # Analysis, while forcing an unexpected persistence-layer exception.
    page = client.get("/upload")
    assert page.status_code == 200 and b'action="/upload"' in page.data
    payload = make_video(tmp_path / "route-diagnostic.avi")
    app.config["MONGO_URI"] = "mongodb+srv://diagnostic-user:diagnostic-secret@invalid.example/db"
    monkeypatch.setattr(video_routes, "create_video", lambda _document: (_ for _ in ()).throw(
        RuntimeError("injected upload persistence failure: " + app.config["MONGO_URI"])))

    response = client.post("/upload", data={
        "csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "route-diagnostic.avi"),
    }, content_type="multipart/form-data")

    assert response.status_code == 500
    assert b"injected upload persistence failure" not in response.data
    assert b"Traceback" not in response.data
    request_id = response.headers.get("X-Request-ID")
    assert request_id and f"Reference: {request_id}".encode() in response.data
    assert f"Unhandled request exception request_id={request_id} (RuntimeError) at endpoint videos.upload" in caplog.text
    assert "injected upload persistence failure" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "diagnostic-secret" not in caplog.text
    assert "<redacted MongoDB URI>" in caplog.text
    assert list((tmp_path / "uploads").iterdir()) == []


def test_startup_identifies_ping_authentication_failure_without_credentials(tmp_path, caplog):
    class Admin:
        def command(self, _name):
            raise OperationFailure("Authentication failed", 18)
    class FakeClient:
        admin = Admin()
        def __getitem__(self, _name):
            return object()
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": FakeClient(),
                              "MONGO_DB_NAME": "isolated_test", "MONGO_PING_ON_STARTUP": True,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_ping_ok"] is False
    assert "startup ping failed (OperationFailure, code=18)" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_atlas_dns_failure_is_explained_and_traceback_redacts_connection_uri(tmp_path, monkeypatch, caplog):
    import app.db as database_module
    from pymongo.errors import ConfigurationError

    fake_uri = "mongodb+srv://diagnostic-user:diagnostic-secret@cluster.invalid/db"
    def fail_client(*_args, **_kwargs):
        raise ConfigurationError("DNS query timed out for " + fake_uri)
    monkeypatch.setattr(database_module, "MongoClient", fail_client)

    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": fake_uri,
                              "MONGO_DB_NAME": "isolated_dns_test", "MONGO_TIMEOUT_MS": 20,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_client"] is None
    assert "PyMongo could not resolve" in application.extensions["mongo_error"]
    assert "Traceback (most recent call last)" in caplog.text
    assert "diagnostic-secret" not in caplog.text
    assert "<redacted MongoDB URI>" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_startup_identifies_index_permission_failure_after_successful_ping(tmp_path, caplog):
    class Admin:
        def command(self, _name):
            return {"ok": 1}
    class Collection:
        def create_index(self, *_args, **_kwargs):
            raise OperationFailure("not authorized", 13)
    class FakeDatabase:
        videos = Collection()
    class FakeClient:
        admin = Admin()
        def __getitem__(self, _name):
            return FakeDatabase()
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": FakeClient(),
                              "MONGO_DB_NAME": "isolated_test", "MONGO_PING_ON_STARTUP": True,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_ping_ok"] is True
    assert application.extensions["mongo_available"] is False
    assert "startup index creation failed (OperationFailure, code=13)" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_missing_or_example_mongodb_uri_keeps_public_pages_available(tmp_path):
    for uri in ("", "paste_your_atlas_connection_string_here",
                "mongodb+srv://<db_username>:<db_password>@cluster0.xxxxx.mongodb.net/"):
        app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": uri,
                          "MONGO_DB_NAME": "unconfigured_test",
                          "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
        client = app.test_client()
        assert app.extensions["mongo_client"] is None
        assert client.get("/").status_code == 200
        response = client.get("/dashboard")
        assert response.status_code == 503
        assert b"replace the value after MONGO_URI=" in response.data
        assert b"MongoDB Atlas" in response.data
        app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_repository_create_read_update_and_dashboard(app):
    with app.app_context():
        record_id = create_video(video_doc())
        assert isinstance(record_id, ObjectId)
        assert get_video(str(record_id))["original_filename"] == "clip.mp4"
        assert update_video(str(record_id), {"processing_status": "complete", "duration": 12.5,
            "face_analysis": {"status": "complete", "face_observations": 4}})
        stats = dashboard_stats()
        assert (stats["videos"], stats["done"], stats["failed"], stats["events"]) == (1, 1, 0, 0)
        assert (stats["face_videos"], stats["face_observations"]) == (1, 4)


def test_safe_objectid_validation():
    assert object_id("not-an-object-id") is None
    assert object_id("507f1f77bcf86cd799439011") == ObjectId("507f1f77bcf86cd799439011")


def test_operation_failure_messages_distinguish_auth_and_permissions():
    auth_message = _friendly_connection_error(OperationFailure("Authentication failed", 18))
    permission_message = _friendly_connection_error(OperationFailure("not authorized", 13))
    assert "authentication failed" in auth_message.lower()
    assert "permission" in permission_message.lower()


def test_fall_heuristic_requires_a_transition_and_persistence():
    class Tensor:
        def __init__(self, values): self.values = np.asarray(values)
        def cpu(self): return self
        def numpy(self): return self.values
        def __len__(self): return len(self.values)
    class Keypoints:
        def __init__(self, points): self.xy = Tensor([points]); self.conf = Tensor([[.9] * 17])
    class Result:
        def __init__(self, points): self.keypoints = Keypoints(points)
        def plot(self): return np.zeros((100, 100, 3), dtype=np.uint8)
    class PoseModel:
        def __init__(self, frames): self.frames = iter(frames)
        def predict(self, frame, verbose=False): return [Result(next(self.frames))]
    def pose(center, horizontal=False):
        pts = np.zeros((17, 2), dtype=float)
        if horizontal:
            pts[5], pts[6], pts[11], pts[12] = [35, center], [65, center], [42, center+3], [72, center+3]
        else:
            pts[5], pts[6], pts[11], pts[12] = [45, center-20], [55, center-20], [45, center+20], [55, center+20]
        return pts.tolist()
    detector = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    detector.model = PoseModel([pose(40), pose(78, True), pose(78, True)])
    detector.previous = None; detector.track_center = None; detector.upright_seen = False; detector.upright_timestamp = None; detector.horizontal_since = None
    detector.drop_seen_at = None; detector.drop_fraction = None; detector.drop_start_at = None; detector.emitted = False
    detector.transition_window_seconds = 1.5; detector.minimum_vertical_drop = .12
    detector.minimum_horizontal_persistence = .6; detector.candidate_score_threshold = .55
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.inspect(frame, 0)[1] is None
    assert detector.inspect(frame, 1)[1] is None
    event = detector.inspect(frame, 2)[1]
    assert event["event_type"] == "potential_fall"
    assert event["confidence"] is None and 0 <= event["anomaly_score"] <= 1
    assert event["start_timestamp_seconds"] == 0 and event["end_timestamp_seconds"] == 2
    assert event["details"]["previously_upright"] is True

    static = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    static.model = PoseModel([pose(80, True), pose(80, True), pose(80, True)])
    static.previous = None; static.track_center = None; static.upright_seen = False; static.upright_timestamp = None; static.horizontal_since = None
    static.drop_seen_at = None; static.drop_fraction = None; static.drop_start_at = None; static.emitted = False
    static.transition_window_seconds = 1.5; static.minimum_vertical_drop = .12
    static.minimum_horizontal_persistence = .6; static.candidate_score_threshold = .55
    assert all(static.inspect(frame, t)[1] is None for t in (0, 1, 2))


def test_object_detector_normalizes_supported_classes_and_temporary_tracks(tmp_path, monkeypatch):
    import app.services.object_detector as object_module
    model_path = tmp_path / "local-coco.pt"
    model_path.write_bytes(b"test model marker")
    names = {0: "person", 1: "bicycle", 2: "car", 24: "backpack"}
    boxes = types.SimpleNamespace(
        xyxy=np.asarray([[1, 2, 30, 50], [10, 10, 40, 40], [20, 20, 30, 30], [0, 0, 1, 1]], dtype=float),
        cls=np.asarray([0, 2, 24, 1]), conf=np.asarray([.91, .82, .77, .99]),
        id=np.asarray([12, 4, -1, 2]),
    )
    class Result:
        pass
    Result.names = names
    Result.boxes = boxes
    class FakeYOLO:
        def __init__(self, _path): self.calls = []; self.names = names
        def track(self, frame, **kwargs):
            self.calls.append(kwargs)
            return [Result()]
    monkeypatch.setattr(object_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    detector = YoloObjectTracker(str(model_path), confidence=.3)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    observations = detector.detect(frame, 1.25, 25)
    assert [row["label"] for row in observations] == ["person", "car", "backpack", "bicycle"]
    assert observations[0]["timestamp_seconds"] == 1.25 and observations[0]["frame_number"] == 25
    assert observations[0]["confidence"] == .91 and observations[0]["track_id"] == 12
    assert observations[1]["track_id"] == 4 and observations[2]["track_id"] is None
    assert observations[3]["track_id"] == 2
    assert detector.model.calls[0]["persist"] is True
    assert detector.model.calls[0]["tracker"] == "bytetrack.yaml"
    assert detector.model.calls[0]["classes"] == [0, 1, 2, 24]
    annotated = detector.annotate(frame, observations)
    assert np.any(annotated != frame) and np.all(frame == 0)


def test_object_detector_reports_missing_configuration_without_downloading(tmp_path, monkeypatch):
    import app.services.object_detector as object_module
    assert object_detector_status()["status"] == "not_configured"
    missing = tmp_path / "missing.pt"
    with pytest.raises(ObjectDetectorUnavailable, match="existing local"):
        YoloObjectTracker(str(missing))
    model_path = tmp_path / "available.pt"
    model_path.write_bytes(b"test marker")
    monkeypatch.setattr(object_module.importlib.util, "find_spec", lambda _name: None)
    assert object_detector_status(str(model_path))["status"] == "unavailable"
    with pytest.raises(ObjectDetectorUnavailable, match="requirements-ai"):
        YoloObjectTracker(str(model_path))


def test_pose_model_alias_does_not_trigger_automatic_download(monkeypatch):
    from app.services.fall_detector import DetectorUnavailable, detector_status as pose_status
    import app.services.fall_detector as pose_module
    monkeypatch.setattr(pose_module.importlib.util, "find_spec", lambda _name: object())
    assert pose_status("yolo26n-pose.pt")["status"] == "unavailable"
    with pytest.raises(DetectorUnavailable, match="not found"):
        YoloPoseFallDetector("yolo26n-pose.pt")


def test_object_sampling_summary_and_csv_use_actual_structured_results(tmp_path):
    source = tmp_path / "sampled.avi"
    make_video(source)
    class FixtureObjectDetector:
        METHOD = "fixture detector"
        confidence = .25
        def detect(self, _frame, timestamp, frame_number):
            return [{"event_type": "object_detection", "label": "person",
                     "timestamp_seconds": timestamp, "frame_number": frame_number,
                     "confidence": .9, "anomaly_score": None, "track_id": 3,
                     "detection_method": self.METHOD,
                     "details": {"bbox_xyxy": [1, 2, 20, 40]}}]
        def annotate(self, frame, _observations): return frame
    result = analyze_video(str(source), .2, 30, object_detector=FixtureObjectDetector())
    assert result["object_analysis"]["frames_sampled"] == 5
    assert result["object_analysis"]["counts_by_class"] == {"person": 5}
    assert result["object_analysis"]["person_track_count_estimate"] == 1
    assert [event["timestamp_seconds"] for event in result["events"]] == [0, .2, .4, .6, .8]
    summary = build_analysis_summary(result, .2)
    assert "5 frame observations" in summary and "not unique identities" in summary
    video = {"_id": ObjectId(), "original_filename": "sampled.avi", "processing_status": "complete",
             "analysis_summary": summary, "analysis_settings": {"sample_every_seconds": .2}}
    csv_text = video_report_csv(video, result["events"])
    assert "object_detection" in csv_text and "person" in csv_text and "0.9" in csv_text


def test_background_worker_persists_only_detector_emitted_events(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    model_marker = tmp_path / "configured-model.pt"
    model_marker.write_bytes(b"test marker")
    app.config["POSE_MODEL_PATH"] = str(model_marker)
    class FixtureDetector:
        METHOD = "synthetic detector fixture"
        def inspect(self, frame, timestamp):
            return frame, {"event_type": "potential_fall", "timestamp_seconds": timestamp,
                           "confidence": None, "anomaly_score": .8, "detection_method": self.METHOD,
                           "metadata": {"source": "synthetic test fixture"}}
        def inspect_with_evidence(self, frame, timestamp):
            overlay, event = self.inspect(frame, timestamp)
            evidence = {"timestamp_seconds": timestamp, "person_count": 1,
                        "people": [{"keypoints": {"left_shoulder": {"x": 12.0, "y": 10.0,
                                                                        "confidence": .9}}}],
                        "transition_status": "candidate_emitted", "model_name": "fixture-pose",
                        "model_version": "test"}
            return overlay, event, evidence
    monkeypatch.setattr(video_routes, "YoloPoseFallDetector", lambda _path, _settings=None: FixtureDetector())
    payload = make_video(tmp_path / "pose-test.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "pose-test.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["analysis_mode"] == "pose_heuristic"
    event = app.extensions["mongo_db"].detection_events.find_one({"video_id": ObjectId(video_id)})
    assert event["event_type"] == "potential_fall" and event["confidence"] is None
    assert event["anomaly_score"] == .8 and event["metadata"]["source"] == "synthetic test fixture"
    assert row["pose_analysis"]["status"] == "complete"
    pose_evidence = app.extensions["mongo_db"].pose_detection_frames.find_one({"video_id": ObjectId(video_id)})
    assert pose_evidence["timestamp_seconds"] == 0
    assert pose_evidence["people"][0]["keypoints"]["left_shoulder"]["confidence"] == .9
    results = client.get(f"/results/{video_id}")
    assert b"synthetic detector fixture" in results.data
    assert b"Inspect sampled pose evidence" in results.data
    if row.get("processed_filename"):
        assert b"Annotated analysis output" in results.data


def test_processed_path_is_restricted_to_generated_name(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    name = "507f1f77bcf86cd799439011_annotated.mp4"
    (root / name).write_bytes(b"video")
    assert safe_processed_path({"processed_filename": name}, str(root)) == str((root / name).resolve())
    outside = tmp_path / "private.txt"
    outside.write_text("keep")
    assert safe_processed_path({"processed_filename": str(outside)}, str(root)) is None


def test_missing_configured_pose_model_is_reported_as_partial_success(client, app, tmp_path):
    app.config["POSE_MODEL_PATH"] = str(tmp_path / "missing-weights.pt")
    payload = make_video(tmp_path / "missing-model.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "missing-model.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["pose_analysis"]["status"] == "unavailable"
    assert "weights" in row["pose_analysis"]["error_message"].lower()
    assert row["analysis_mode"] == "validation_only"


def test_safe_video_path_requires_generated_internal_name(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    good_name = "a" * 32 + ".mp4"
    good = folder / good_name
    good.write_bytes(b"video")
    assert safe_video_path({"stored_filename": good_name, "file_path": good_name}, str(folder)) == str(good.resolve())
    outside = tmp_path / "important.txt"
    outside.write_text("keep")
    assert safe_video_path({"stored_filename": str(outside), "file_path": str(outside)}, str(folder)) is None
    assert outside.read_text() == "keep"


def test_history_search_filter_event_and_csv(client, app):
    with app.app_context():
        one = create_video(video_doc("fall-study.mp4", "complete"))
        create_video(video_doc("scene-two.avi", "failed"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": one, "event_type": "prototype_test_event",
            "timestamp_seconds": 3.0, "confidence": None, "anomaly_score": None,
            "detection_method": "test-fixture", "created_at": datetime.now(timezone.utc)})
        stats = dashboard_stats()
        assert stats["events"] == 1 and stats["categories"][0]["category"] == "prototype_test_event"
        videos, events = report_data()
        assert len(videos) == 2 and len(events) == 1
        assert report_data("2099-01-01", "2099-01-02") == ([], [])
        assert list_videos("scene", "failed")[1] == 1
    response = client.get("/history?q=fall&status=complete")
    assert response.status_code == 200 and b"fall-study.mp4" in response.data and b"scene-two.avi" not in response.data
    csv_response = client.get("/reports.csv")
    assert csv_response.status_code == 200 and b"prototype_test_event" in csv_response.data


def test_delete_removes_only_video_and_its_events(app):
    with app.app_context():
        target = create_video(video_doc("remove.mp4"))
        keep = create_video(video_doc("keep.mp4"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": target, "event_type": "x"})
        db.detection_events.insert_one({"video_id": keep, "event_type": "y"})
        deleted = delete_video(str(target))
        assert deleted["_id"] == target
        assert get_video(str(target)) is None
        assert db.detection_events.count_documents({"video_id": target}) == 0
        assert get_video(str(keep)) is not None
        assert db.detection_events.count_documents({"video_id": keep}) == 1


def test_delete_route_removes_owned_file(client, app, tmp_path):
    payload = make_video(tmp_path / "delete-me.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "delete-me.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    record = app.extensions["mongo_db"].videos.find_one({})
    stored_path = os.path.join(app.config["UPLOAD_DIR"], record["stored_filename"])
    assert os.path.isfile(stored_path)
    deleted = client.post(f"/history/{video_id}/delete", data={"csrf_token": csrf(client)})
    assert deleted.status_code == 302
    assert not os.path.exists(stored_path)
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0


def test_csrf_and_unsupported_upload(client):
    assert client.post("/history/no/delete").status_code == 400
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "bad.exe")}, content_type="multipart/form-data", follow_redirects=True)
    assert b"Supported extensions" in response.data


def test_upload_saves_processing_result_and_database_error_cleanup(client, app, tmp_path, monkeypatch):
    payload = make_video(tmp_path / "small.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "sample.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = wait_for_status(app, video_id)
    result_page = client.get(response.headers["Location"])
    assert result_page.status_code == 200 and b"Video validation complete" in result_page.data
    assert b"OpenCV validation passed" in result_page.data
    assert b"Face detection" in result_page.data
    assert saved["file_size"] == len(payload)
    assert saved["validation_status"] == "valid"
    assert (saved["width"], saved["height"], saved["frame_count"]) == (64, 48, 10)
    assert saved["face_analysis"]["status"] == "complete"
    assert saved["face_analysis"]["frames_sampled"] > 0
    original = client.get(f"/media/{video_id}")
    download = client.get(f"/media/{video_id}/download")
    assert original.status_code == 200 and original.data == payload
    assert download.status_code == 200 and download.data == payload
    assert "attachment" in download.headers["Content-Disposition"]
    assert saved.get("processed_filename")
    annotated = client.get(f"/processed/{video_id}")
    annotated_download = client.get(f"/processed/{video_id}/download")
    assert annotated.status_code == 200 and annotated.data
    assert annotated.data != original.data
    assert annotated_download.status_code == 200
    assert "attachment" in annotated_download.headers["Content-Disposition"]
    assert app.extensions["mongo_db"].face_detection_frames.count_documents({"video_id": ObjectId(video_id)}) > 0
    restarted = create_app({"TESTING": True, "SECRET_KEY": "test-secret",
        "MONGO_CLIENT": app.extensions["mongo_client"], "MONGO_DB_NAME": "test_human_anomaly",
        "MONGO_PING_ON_STARTUP": False, "UPLOAD_DIR": app.config["UPLOAD_DIR"],
        "PROCESSED_DIR": app.config["PROCESSED_DIR"]})
    assert restarted.test_client().get(f"/media/{video_id}").data == payload
    restarted.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    assert app.extensions["mongo_db"].videos.count_documents({"processing_status": "complete"}) == 1
    status = client.get(f"/api/videos/{video_id}/status")
    assert status.status_code == 200 and status.json["status"] == "complete"
    job = app.extensions["mongo_db"].processing_jobs.find_one({"video_id": ObjectId(video_id)})
    assert job["progress_percent"] == 100 and job["completed_at"] is not None

    import app.routes.videos as video_routes
    monkeypatch.setattr(video_routes, "create_video", lambda _doc: (_ for _ in ()).throw(DatabaseUnavailable("offline")))
    before = set(os.listdir(app.config["UPLOAD_DIR"]))
    failed = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "orphan.avi")}, content_type="multipart/form-data")
    assert failed.status_code == 503
    assert b"OpenCV validated the video" in failed.data
    assert b"no video record or analysis job was saved" in failed.data
    app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    assert set(os.listdir(app.config["UPLOAD_DIR"])) == before


def test_invalid_video_is_rejected_before_database_or_job_creation(client, app, tmp_path):
    before_files = set(os.listdir(app.config["UPLOAD_DIR"]))
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "broken.mp4")},
        content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"could not open this video" in response.data.lower() or b"could not decode" in response.data.lower()
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0
    assert app.extensions["mongo_db"].processing_jobs.count_documents({}) == 0
    assert set(os.listdir(app.config["UPLOAD_DIR"])) == before_files


def test_video_validator_checks_decodable_frame_and_extracts_metadata(tmp_path):
    video_path = tmp_path / "readable.avi"
    make_video(video_path)
    details = validate_video_file(str(video_path))
    assert details["frame_rate"] == 10
    assert details["frame_count"] == 10
    assert details["duration"] == 1
    assert (details["width"], details["height"]) == (64, 48)

    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_bytes(b"not a video")
    with pytest.raises(VideoValidationError):
        validate_video_file(str(corrupt_path))

    with pytest.raises(VideoValidationError):
        validate_video_file(str(tmp_path / "missing.mp4"))


def test_upload_persists_object_observations_tracks_summary_and_report(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    model_path = tmp_path / "local-object-model.pt"
    model_path.write_bytes(b"test marker")
    app.config["OBJECT_MODEL_PATH"] = str(model_path)
    app.config["OBJECT_MODEL_CONFIDENCE"] = .3
    class FixtureObjectDetector:
        METHOD = "fixture object detector"
        confidence = .3
        def __init__(self, *_args): pass
        def detect(self, _frame, timestamp, frame_number):
            return [
                {"event_type": "object_detection", "label": "person", "timestamp_seconds": timestamp,
                 "frame_number": frame_number, "confidence": .91, "anomaly_score": None, "track_id": 8,
                 "detection_method": self.METHOD, "details": {"bbox_xyxy": [1, 2, 20, 40]}},
                {"event_type": "object_detection", "label": "car", "timestamp_seconds": timestamp,
                 "frame_number": frame_number, "confidence": .82, "anomaly_score": None, "track_id": None,
                 "detection_method": self.METHOD, "details": {"bbox_xyxy": [22, 8, 48, 30]}},
            ]
        def annotate(self, frame, _observations): return frame.copy()
    monkeypatch.setattr(video_routes, "YoloObjectTracker", FixtureObjectDetector)
    payload = make_video(tmp_path / "object-upload.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "object-upload.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = wait_for_status(app, video_id)
    assert saved["object_analysis"]["status"] == "complete"
    assert saved["object_analysis"]["counts_by_class"] == {"car": 10, "person": 10}
    assert saved["object_analysis"]["person_track_count_estimate"] == 1
    assert saved["event_count"] == 0 and saved["object_detection_count"] == 20
    object_event = app.extensions["mongo_db"].detection_events.find_one(
        {"video_id": ObjectId(video_id), "label": "person"})
    assert object_event["confidence"] == .91 and object_event["anomaly_score"] is None
    assert object_event["timestamp_seconds"] == 0 and object_event["frame_number"] == 0
    assert object_event["track_id"] == 8 and object_event["details"]["bbox_xyxy"] == [1, 2, 20, 40]
    assert "car: 10 frame observations" in saved["analysis_summary"]
    assert "person: 10 frame observations" in saved["analysis_summary"]
    page = client.get(response.headers["Location"])
    assert page.status_code == 200 and b"person tracking" in page.data
    assert b"event-type-filter" in page.data and b"event-track-filter" in page.data
    assert b"event-time-from" in page.data and b"timeline-event" in page.data
    report = client.get(f"/results/{video_id}/report.csv")
    assert report.status_code == 200 and b"object_detection" in report.data and b"person" in report.data
    dashboard = client.get("/dashboard")
    assert b"<strong>1</strong>" in dashboard.data
    with app.app_context():
        stats = dashboard_stats()
    assert stats["object_observations"] == 20 and stats["person_track_estimate"] == 1
    assert {item["label"]: item["count"] for item in stats["object_categories"]} == {"car": 10, "person": 10}
    assert app.extensions["mongo_db"].detection_events.count_documents(
        {"video_id": ObjectId(video_id), "event_type": {"$ne": "object_detection"}}) == 0


def test_opencv_face_detector_loads_bundled_cascade_and_handles_empty_frame():
    assert face_detector_status()["ready"] is True
    detector = OpenCVFaceDetector()
    assert detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)) == []
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_face_detector_missing_config_is_reported_without_download(tmp_path):
    missing = str(tmp_path / "not-present.xml")
    assert face_detector_status(missing)["ready"] is False
    with pytest.raises(FaceDetectorUnavailable, match="FACE_CASCADE_PATH"):
        OpenCVFaceDetector(missing)


def test_missing_face_cascade_does_not_block_other_video_processing(client, app, tmp_path):
    app.config["FACE_CASCADE_PATH"] = str(tmp_path / "missing-cascade.xml")
    payload = make_video(tmp_path / "missing-cascade.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "missing-cascade.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["face_analysis"]["status"] == "unavailable"
    assert row["analysis_mode"] == "validation_only"


def test_face_boxes_support_multiple_results_and_are_annotated():
    class TwoFaces:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[4, 5, 20, 21], [40, 10, 22, 22]])
    detector = OpenCVFaceDetector.__new__(OpenCVFaceDetector)
    detector.cascade = TwoFaces()
    frame = np.zeros((80, 90, 3), dtype=np.uint8)
    boxes = detector.detect(frame)
    assert len(boxes) == 2 and boxes[0]["width"] == 20
    marked = detector.annotate(frame, boxes)
    assert np.any(marked != frame) and np.all(frame == 0)


def test_face_detector_initialization_error_does_not_fail_video_processing(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    class BrokenFaceDetector:
        METHOD = OpenCVFaceDetector.METHOD
        def __init__(self, _path=""):
            raise RuntimeError("private OpenCV detail")
    monkeypatch.setattr(video_routes, "OpenCVFaceDetector", BrokenFaceDetector)
    payload = make_video(tmp_path / "face-init-failure.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "face-init-failure.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["face_analysis"]["status"] == "unavailable"
    assert "private OpenCV detail" not in str(row["face_analysis"])


def test_annotated_writer_failure_does_not_break_original_video_processing(client, app, tmp_path, monkeypatch):
    payload = make_video(tmp_path / "writer-failure.avi")
    monkeypatch.setattr(cv2, "VideoWriter", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(cv2.error("writer unavailable")))
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "writer-failure.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert not row.get("processed_filename")
    assert client.get(f"/media/{video_id}").data == payload


def test_failed_video_retry_claims_one_worker_and_persists_new_job(client, app, tmp_path):
    payload = make_video(tmp_path / "retry-source.avi")
    stored = "a" * 32 + ".avi"
    path = os.path.join(app.config["UPLOAD_DIR"], stored)
    with open(path, "wb") as output:
        output.write(payload)
    doc = video_doc("retry-source.avi", "failed")
    doc.update({"stored_filename": stored, "file_path": stored, "file_size": len(payload)})
    with app.app_context():
        video_id = create_video(doc)
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": video_id, "event_type": "partial_attempt"})
        db.face_detection_frames.insert_one({"video_id": video_id, "timestamp_seconds": 99, "face_count": 1})
    response = client.post(f"/history/{video_id}/retry", data={"csrf_token": csrf(client)})
    assert response.status_code == 302
    row = wait_for_status(app, str(video_id))
    assert row["processing_status"] == "complete"
    assert app.extensions["mongo_db"].processing_jobs.count_documents({"video_id": video_id}) == 1
    old_event = app.extensions["mongo_db"].detection_events.find_one(
        {"video_id": video_id, "event_type": "partial_attempt"})
    old_frame = app.extensions["mongo_db"].face_detection_frames.find_one(
        {"video_id": video_id, "timestamp_seconds": 99})
    assert old_event["superseded"] is True and old_frame["superseded"] is True
    assert app.extensions["mongo_db"].detection_events.count_documents(
        {"video_id": video_id, "superseded": {"$ne": True}}) == 0
    with app.app_context():
        assert dashboard_stats()["events"] == 0
        assert all(event.get("event_type") != "partial_attempt" for event in report_data()[1])
    assert b"partial_attempt" not in client.get("/reports.csv").data
    client.post(f"/history/{video_id}/retry", data={"csrf_token": csrf(client)})
    assert app.extensions["mongo_db"].processing_jobs.count_documents({"video_id": video_id}) == 1


def test_startup_marks_incomplete_jobs_interrupted_in_isolated_database(tmp_path):
    mongo = mongomock.MongoClient()
    db = mongo["interruption_test"]
    video_id = ObjectId()
    db.videos.insert_one({"_id": video_id, "processing_status": "processing"})
    db.processing_jobs.insert_one({"video_id": video_id, "status": "running"})
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": mongo,
        "MONGO_DB_NAME": "interruption_test", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    interrupted = db.videos.find_one({"_id": video_id})
    assert interrupted["processing_status"] == "interrupted"
    assert interrupted["pose_analysis"]["status"] == "interrupted"
    assert interrupted["object_analysis"]["status"] == "interrupted"
    job = db.processing_jobs.find_one({"video_id": video_id})
    assert job["status"] == "interrupted" and job["completed_at"] is not None
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    mongo.close()


def test_face_frames_are_removed_with_only_their_video(app):
    with app.app_context():
        target = create_video(video_doc("face-target.mp4"))
        other = create_video(video_doc("face-other.mp4"))
        db = app.extensions["mongo_db"]
        db.face_detection_frames.insert_one({"video_id": target, "timestamp_seconds": 1, "face_count": 1})
        db.face_detection_frames.insert_one({"video_id": other, "timestamp_seconds": 2, "face_count": 0})
        delete_video(str(target))
        assert db.face_detection_frames.count_documents({"video_id": target}) == 0
        assert db.face_detection_frames.count_documents({"video_id": other}) == 1


def test_empty_video_upload_is_rejected_before_mongodb_insert(client, app):
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b""), "empty.mp4")}, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200 and b"file is empty" in response.data
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0
