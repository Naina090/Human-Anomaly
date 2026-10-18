# Viva preparation

**What does the project do?** It is a browser-based Flask website for uploading local videos, tracking analysis jobs, reviewing results and reporting stored records through MongoDB.

**What activity is detected?** Only a potential-fall transition, and only when optional COCO pose weights are configured. The base install checks whether OpenCV can decode the video and records metadata; it does not predict activities.

**How does the fall prototype work?** It estimates pose keypoints and requires earlier upright evidence, a rapid body-center drop, and a horizontal torso persisting across samples. A single low or horizontal posture is not sufficient. The score is an explainable heuristic formula, not trained-model confidence.

**Has the detector been validated?** No. The repository contains synthetic unit tests for rule behavior, not a labeled real-video evaluation. It makes no accuracy claim, and events can be missed or falsely flagged.

**Why use pose estimation instead of ordinary object detection?** A person box alone does not describe posture change over time. Pose provides body keypoints, which the temporal rule uses. Pose outputs still do not prove that a fall occurred.

**What is stored in MongoDB?** Video metadata, job state/progress and detector events that were actually emitted. Video bytes remain on disk.

**Why process in a thread pool?** It keeps Flask requests responsive for a local demo. The queue is bounded. It is process-local, so queued work is lost if Flask stops; production deployment would need a durable worker queue.

**What does zero events mean?** No event documents were emitted or retained. It is not evidence that the video is normal.

**How are event times reviewed?** The result page shows seconds from the source clip; selecting a timestamp seeks the original HTML video player.

**Does it recognize identity or dangerousness?** No. It performs neither identity recognition nor danger assessment.

**What are remaining limitations?** Heuristic not validated, crowded-scene pose association can switch people, optional weights/dependencies need setup, output codecs vary, no user login, and the local application is not hardened for public deployment.
