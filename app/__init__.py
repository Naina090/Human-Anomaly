import os
import secrets
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from flask import Flask, render_template, session, request, abort, g
from werkzeug.exceptions import HTTPException
from config import Config
from .db import init_database, DatabaseUnavailable
from .logging_utils import sanitized_traceback
from .routes import register_routes


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    # This is a locally edited college project. Reload changed Jinja templates
    # and avoid showing a browser-cached pre-update dashboard.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    if test_config:
        app.config.update(test_config)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["PROCESSED_DIR"], exist_ok=True)
    init_database(app)
    if app.extensions.get("mongo_available"):
        try:
            with app.app_context():
                from app.services.repository import mark_incomplete_jobs_interrupted
                mark_incomplete_jobs_interrupted()
        except DatabaseUnavailable:
            logging.getLogger(__name__).warning("Could not recover interrupted jobs because MongoDB is unavailable.")
    workers = max(1, int(app.config.get("PROCESSING_WORKERS", 1)))
    app.extensions["processing_executor"] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-analysis")
    app.extensions["processing_slots"] = BoundedSemaphore(workers + max(0, int(app.config.get("PROCESSING_QUEUE_SIZE", 3))))
    register_routes(app)

    @app.before_request
    def assign_request_id():
        # A short per-request token connects a browser error page to the full
        # server-side traceback without logging the URL query or request body.
        g.request_id = secrets.token_hex(8)

    @app.before_request
    def check_csrf():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(24)
        if request.method == "POST" and request.form.get("csrf_token") != session["csrf_token"]:
            abort(400)

    @app.after_request
    def avoid_stale_local_pages(response):
        if response.mimetype == "text/html" or request.endpoint == "static":
            response.headers["Cache-Control"] = "no-store"
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.context_processor
    def csrf_context():
        model_path = app.config.get("POSE_MODEL_PATH", "")
        return {"csrf_token": session["csrf_token"],
                "detector_configured": bool(model_path and os.path.isfile(model_path))}

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", message="The video exceeds the configured upload limit."), 413

    @app.errorhandler(DatabaseUnavailable)
    def database_unavailable(error):
        return render_template(
            "error.html",
            error_label="MONGODB CONNECTION",
            error_title="MongoDB is temporarily unavailable.",
            message=str(error),
        ), 503

    @app.errorhandler(Exception)
    def unexpected_error(error):
        # Werkzeug HTTP errors (404, 405, etc.) should keep their normal status.
        if isinstance(error, HTTPException):
            return error
        # Log the full server-side traceback for diagnosis, but never include the
        # request body, uploaded content, query string, or MongoDB credentials.
        formatted = sanitized_traceback(error, app.config)
        app.logger.error(
            "Unhandled request exception request_id=%s (%s) at endpoint %s; traceback follows:\n%s",
            getattr(g, "request_id", "unknown"), type(error).__name__, request.endpoint or "unknown", formatted,
        )
        return render_template(
            "error.html",
            message="The server could not complete this request. Share the reference below with your project support person.",
            request_id=getattr(g, "request_id", "unknown"),
        ), 500

    return app
