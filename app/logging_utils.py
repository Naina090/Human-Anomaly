"""Server-side exception formatting with project-secret and local-path redaction."""
import re
import traceback


def sanitized_traceback(error, config):
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    uri = str(config.get("MONGO_URI") or "")
    if uri:
        text = text.replace(uri, "<redacted MongoDB URI>")
    text = re.sub(r"(?i)(mongodb(?:\+srv)?://[^:/@\s]+:)[^@/\s]+@", r"\1<redacted>@", text)
    for key in ("UPLOAD_DIR", "PROCESSED_DIR", "POSE_MODEL_PATH", "OBJECT_MODEL_PATH"):
        configured_path = str(config.get(key) or "")
        if configured_path:
            text = text.replace(configured_path, f"<{key}>")
    return text
