# Web-Based Human Anomaly Detection System

Sentinel Activity Lab is a browser-based Python Flask research prototype. Flask and Jinja serve the landing page, dashboard, upload, results, history, reports and methodology pages. Uploaded video files stay in local storage; MongoDB stores their metadata, processing jobs and only genuine detector event records. There is no mobile app, React frontend, or Node build step.

## What works

- MongoDB persistence through a shared PyMongo client: video records, job status, event history, dashboard queries, deletion and CSV reporting.
- Local upload with internal random filenames, extension/size/duration constraints, preview, private media route and safe deletion.
- Bounded in-process background queue with progress/status polling. This is appropriate for a single-user local demonstration; a process restart loses queued work, and it is not a multi-server worker system.
- OpenCV streaming decode/metadata checks. Without model weights, a completed run is explicitly `validation_only` and creates no anomaly events.
- Optional pose-based `potential_fall` transition prototype (see below). Annotated MP4 output is generated when OpenCV's video writer supports the selected codec.
- Searchable history, status filter, safe record deletion, actual database-backed counts and date-filtered CSV reports.
- CSRF-protected state-changing forms. There is no user account/authentication layer; keep this demo on a trusted local machine.

## Defined AI activity and limitations

The optional detector uses an Ultralytics COCO-17 pose model to estimate keypoints. It does **not** use generic person detection as a fall classifier. The rule requires (a) prior upright torso evidence, (b) a rapid body-center vertical drop within 1.5 seconds, and (c) a horizontal torso persisting for at least 0.6 seconds. Its heuristic score is `0.5 × min(1, drop_fraction / 0.3) + 0.3 × min(1, torso_horizontal_ratio / 2) + 0.2 × min(1, persistence_seconds / 1.2)`; it is an anomaly score, not model confidence. Confidence remains null. The current prototype emits at most one qualifying event per video. It is not clinically or scientifically validated, can miss events or flag other movements, and cannot establish intent, danger, injury, or identity. Fighting and general unusual behavior are not implemented. An empty event list does not mean that a video is normal.

Pose model dependency and weights are optional. Ultralytics documents COCO pose keypoints and pretrained pose weights in its [Pose task guide](https://docs.ultralytics.com/tasks/pose); its licensing page explains the default AGPL-3.0 terms and Enterprise option for qualifying proprietary/commercial use: [Ultralytics licensing](https://www.ultralytics.com/license). Review license terms before distributing a project that bundles the package or weights. The app does not train a model. Model weights can be downloaded separately by Ultralytics at first run, which requires network access; use a local path instead for offline runs. Hardware speed depends on the clip, CPU/GPU and PyTorch build; no performance or accuracy claim is made.

## Requirements and setup (Windows / VS Code)

Install Python 3.10+ and have a MongoDB Atlas cluster or local MongoDB server. Open this folder in VS Code, then run these exact PowerShell commands:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Configure MongoDB Atlas

1. Create/sign in to MongoDB Atlas and create a cluster.
2. Under **Database Access**, create a database user with a strong password. This is distinct from your Atlas login.
3. Under **Network Access**, allow your current IP address. Avoid allowing all IPs for routine use.
4. On the cluster, choose **Connect → Drivers → Python** and copy the connection URI. Put that URI only in `.env`; URL-encode reserved password characters.
5. Open `.env` and set:

```dotenv
SECRET_KEY=use-a-long-private-random-value
MONGO_URI=mongodb+srv://DB_USER:ENCODED_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=human_anomaly_db
```

`.env` is ignored by Git. The first successful connection creates indexes; MongoDB shows collections after their first records are inserted. Upload a short video, then in Atlas open **Browse Collections → human_anomaly_db → videos**. Event rows appear only when the configured detector actually emits an event.

Local server alternative: `MONGO_URI=mongodb://127.0.0.1:27017/`, database name `human_anomaly_db`; start the MongoDB service first.

### Optional pose-model setup

Install PyTorch using the current command recommended by the [official PyTorch install selector](https://pytorch.org/get-started/locally/) for your Windows CPU/CUDA setup, then install the optional integration:

```powershell
python -m pip install -r requirements-ai.txt
```

Use the documented pretrained `yolo26n-pose.pt` pose weights from the Ultralytics Pose task. Set `POSE_MODEL_PATH=yolo26n-pose.pt` in `.env` to allow Ultralytics to download that model on first use, or specify an existing local path such as `POSE_MODEL_PATH=C:\Models\yolo26n-pose.pt`. Restart Flask after changing `.env`. If AI dependencies are not installed or weights cannot load, the processing job is marked failed rather than silently reverting or inventing results. Without a model path, metadata validation remains available.

### Start the browser website

```powershell
python app.py
```

Open http://127.0.0.1:5000. If PowerShell blocks virtual-environment activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MONGO_URI`, `MONGO_DB_NAME` | local URI, `human_anomaly_db` | MongoDB connection/database |
| `SECRET_KEY` | generated per process if omitted | Flask session signature; set a stable private value |
| `MAX_UPLOAD_MB` | 250 | Maximum request size |
| `MAX_DURATION_SECONDS` | 900 | Maximum clip duration |
| `SAMPLE_EVERY_SECONDS` | 1.0 | Frame inference sampling interval |
| `PROCESSING_WORKERS` | 1 | Concurrent jobs in this Flask process |
| `PROCESSING_QUEUE_SIZE` | 3 | Additional queued jobs |
| `POSE_MODEL_PATH` | unset | Optional local pose weight path |
| `UPLOAD_DIR`, `PROCESSED_DIR` | project folders | Local originals and annotated output |

## Collections and architecture

`videos` stores original filename, random internal file reference, upload date, duration, FPS, resolution, processing state/time, analysis method/mode, and an error string. `processing_jobs` stores video reference, queued/running/complete/failed state, progress and timestamps. `detection_events` stores video ObjectId, category, video timestamp, nullable model confidence, optional heuristic score, method and evidence metadata. Large files are not stored in MongoDB.

```mermaid
flowchart LR
 Browser --> Flask[Flask routes and Jinja]
 Flask --> Mongo[(MongoDB: videos, processing_jobs, detection_events)]
 Flask --> Queue[Bounded in-process thread pool]
 Queue --> CV[OpenCV streaming decoder]
 CV --> Pose[Optional YOLO pose model]
 Pose --> Rule[Temporal fall-transition heuristic]
 CV --> Local[(Local uploads and processed files)]
 Rule --> Mongo
 Mongo --> Pages[Results, dashboard, history, reports]
```

## Tests

Tests use an isolated in-memory `mongomock` database; they never connect to or clear the configured database. Run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The suite covers Flask pages, Mongo connection failure, repository persistence, event/report queries, upload and validation errors, async completion, deletion boundaries, ObjectId checks, and the transition heuristic with synthetic keypoints. It does not download weights or verify real-world model performance. A real Atlas server and an installed pose model require separate environment testing.

## Troubleshooting

- **MongoDB unavailable:** confirm `.env` is beside `app.py`, the URI/user/password are valid, Atlas allows your current IP, and the cluster is running. Restart Flask after edits. Error pages do not reveal credentials.
- **Invalid password with special characters:** URL-encode reserved characters in the URI password component.
- **Unreadable video:** use a short MP4 (H.264 is commonly supported) or a codec available in your OpenCV build. Extension alone does not prove the content is a video.
- **Model cannot load:** confirm `requirements-ai.txt`, PyTorch hardware build, model path and weights. Do not commit large weights or secrets.
- **No events:** expected in validation-only mode, or when the heuristic did not observe its transition conditions. It is not evidence that no fall occurred.
- **No annotated preview:** only produced with a configured pose model and a working OpenCV MP4 writer. Original playback remains available.

## Privacy and project submission

Use consented, non-sensitive footage. The local demo has no authentication, account-based authorization, HTTPS deployment, or distributed job recovery; do not expose it to the public internet. Media is served through record IDs and resolved within configured folders. Delete records from history to remove their local files and associated events. The system does not recognize identity or label a person as dangerous.

See [college project report](docs/PROJECT_REPORT.md) and [viva preparation](docs/VIVA_PREPARATION.md). They distinguish implemented behavior from future work; add only sources and experiment results you actually verify.
