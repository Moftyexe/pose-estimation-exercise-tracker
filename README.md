\# Shoulder Press Pose Tracker (MediaPipe + OpenCV)



A real-time shoulder press form tracker built in Python using \*\*MediaPipe Pose\*\* and \*\*OpenCV\*\*.



\## Features

\- Detects left/right elbow angles live

\- Counts reps for each arm

\- Tracks ROM (range of motion), TUT (time under tension), and tempo (deg/sec)

\- Flags asymmetry + form issues (ROM/lockout/tempo)

\- Saves a session summary to a CSV file (`data/shoulder\_press\_log.csv`)



\## Demo

!\[Demo Screenshot](media/demo\_screenshot.png)





\## Requirements

\- Python 3.9+ recommended

\- Webcam



\## Setup

```bash

pip install -r requirements.txt

