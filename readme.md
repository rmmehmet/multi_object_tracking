# 🚗🧍 Multi-Object Tracker — Cars & Pedestrians

**Real-time multi-class object tracking powered by YOLOv26, Deep SORT & OSNet Re-ID**

*Computer vision meets swarm-inspired detection — track what moves, remember who it is*

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![YOLOv26](https://img.shields.io/badge/YOLOv26-Detection-purple?logo=github)
![DeepSORT](https://img.shields.io/badge/Deep_SORT-Re--ID-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

A real-time multi-object tracking pipeline built with **YOLOv26**, **Deep SORT**, and **OSNet** that simultaneously tracks vehicles and pedestrians in video footage. Cars are tracked with an IoU-based tracker featuring motion trails, while pedestrians are tracked using appearance-based re-identification via OSNet embeddings.

---

## ✨ Features

- **Dual-class tracking** — independent tracker instances for persons and vehicles
- **IoU-based car tracker** — lightweight, fast, and effective for vehicle tracking
- **Deep SORT person tracker** — appearance-aware tracking using cosine distance metric
- **OSNet re-identification** — robust feature extraction for pedestrian re-ID across frames
- **Motion trail visualization** — historical bounding box paths rendered for car tracks
- **Unique color-coded IDs** — distinct visual labels per tracked object (`P-{id}` / `C-{id}`)
- **Configurable confidence threshold** — easy filtering of low-confidence detections

---

## 🏗️ Architecture

```
Input Video
    │
    ▼
┌─────────────────────────┐
│     YOLOv26 Detector    │  ← Detects persons (cls=0) & cars (cls=2)
└─────────────────────────┘
         │              │
         ▼              ▼
┌──────────────┐  ┌─────────────────────────────────────┐
│  CarTracker  │  │           PersonTracker              │
│  (IoU-based) │  │  Deep SORT + OSNet Re-ID Encoder    │
└──────────────┘  └─────────────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────────────┐
│   Annotated Output      │  ← Bounding boxes, IDs, trails
│       Video             │
└─────────────────────────┘
```

---

## 📁 Project Structure

```
├── models/
│   ├── yolo26n.pt                  # YOLOv8 weights
│   └── osnet_encoder.py            # OSNet feature extractor
├── trackers/
│   ├── car_tracker.py              # IoU-based car tracker
│   └── person_tracker.py           # Deep SORT person tracker
├── deep_sort/
│   └── deep_sort/                  # Deep SORT library
│       ├── tracker.py
│       ├── nn_matching.py
│       └── detection.py
├── videos/                         # Input videos
├── output/                         # Processed output videos
└── main.py                         # Entry point
```

---

## 🛠️ Requirements

**Python 3.8+**

```bash
pip install ultralytics opencv-python torch torchreid numpy
```

> **Note:** Ensure you have a compatible version of CUDA if you intend to run inference on GPU. The OSNet encoder will automatically detect and use `cuda` if available.

### Deep SORT

Clone and place the Deep SORT library under the project root:

```bash
git clone https://github.com/nwojke/deep_sort.git
```

---

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/your-username/multi-object-tracker.git
cd multi-object-tracker
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Place your input video** under `videos/`

4. **Run the tracker**

```bash
python main.py
```

The annotated output will be saved to `output/output_new_method.mp4`.

---

## ⚙️ Configuration

All key parameters are defined at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Minimum detection confidence score |
| `video_path` | `videos/...` | Path to input video file |
| `video_out_path` | `output/...` | Path to output video file |

**CarTracker** parameters (in `car_tracker.py`):

| Parameter | Default | Description |
|---|---|---|
| `iou_threshold` | `0.4` | Minimum IoU to associate a detection with an existing track |
| `trail_length` | `10` | Number of past frames used to draw the motion trail |

**PersonTracker** parameters (in `person_tracker.py`):

| Parameter | Default | Description |
|---|---|---|
| `max_cosine_distance` | `0.3` | Cosine distance threshold for Re-ID matching |
| `time_since_update` | `10` | Max frames a track can be unmatched before removal |

---

## 🧠 How It Works

### Car Tracking (IoU-based)

The `CarTracker` uses Intersection over Union (IoU) to associate new detections with existing tracks each frame. If the IoU between a detection and an existing track exceeds 0.4, the track is updated; otherwise a new track is created. Historical bounding boxes are stored per track and used to render a motion trail on the output video.

### Person Tracking (Deep SORT + OSNet)

The `PersonTracker` wraps the Deep SORT algorithm with an OSNet-powered appearance encoder. For each detected person, a 128×256 image crop is extracted, preprocessed, and passed through `osnet_x1_0` (pretrained on a large Re-ID dataset) to produce a normalized 512-dimensional feature vector. Deep SORT uses these embeddings alongside Kalman-filter-based motion prediction to maintain stable identities across frames — even through short occlusions.

---

## 📊 Output

- **Persons** are drawn with green bounding boxes labeled `P-{id}`
- **Cars** are drawn with blue bounding boxes labeled `C-{id}` and include a motion trail showing the last 10 positions

---

## 📌 Notes

- The model file `yolo26n.pt` must be placed under `model/` before running.
- `torchreid` will automatically download the pretrained `osnet_x1_0` weights on first run.
- For best performance on long videos or dense scenes, GPU acceleration is strongly recommended.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — YOLOv26
- [Deep SORT](https://github.com/nwojke/deep_sort) — Simple Online and Realtime Tracking with a Deep Association Metric
- [torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid) — Omni-Scale Feature Learning for Person Re-Identification