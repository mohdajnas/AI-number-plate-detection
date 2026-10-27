# AI License & Number Plate Detection System (ANPR)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Deep%20Learning-FF6F00?style=for-the-badge)](https://github.com/JaidedAI/EasyOCR)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An end-to-end, deep learning-powered **Automated Number Plate Recognition (ANPR)** web application. This system integrates a fine-tuned **YOLOv8** model for high-precision license plate localization, an advanced **OpenCV computer vision preprocessing pipeline** (CLAHE adaptive contrast enhancement, bicubic upscaling, edge noise cropping), and **EasyOCR** with regex-based Indian vehicle registration normalization.

---

## Visual Showcase

### 1. Real-Time Detection & Recognition
Accurate plate boundary localization with confidence score and extracted alphanumeric decoding.

<p align="center">
  <img src="assets/detection_preview.png" alt="License Plate Detection Preview" width="850px" style="border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />
</p>

---

### 2. Detection History & Audit Logging
Every processed plate is saved with its cropped image reference, detection confidence, and UTC timestamp into an SQLite datastore.

<p align="center">
  <img src="assets/history_preview.png" alt="Detection History Dashboard" width="850px" style="border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />
</p>

---

### 3. Cropped License Plate Samples (`detected_plates/`)
The system extracts individual plate bounding boxes and applies custom OCR filtering to extract crisp, readable text even from varied lighting and angles.

| Sample Plate Crop | Detected Text | Detection Confidence | Plate Type |
| :---: | :---: | :---: | :---: |
| <img src="assets/plates/plate_sample_1.jpg" width="220px" alt="DL 7CQ 1939" /> | `DL7CQ1939` | **96.5%** | Private Vehicle (White Plate) |
| <img src="assets/plates/plate_sample_2.jpg" width="220px" alt="DL 1C AB 1234" /> | `DL1CAB1234` | **94.6%** | Private Vehicle (White Plate) |
| <img src="assets/plates/plate_sample_3.jpg" width="220px" alt="DL 1Z C 5678" /> | `DL1ZC5678` | **74.2%** | Commercial Vehicle (Yellow Plate) |

---

## Key Features

- **Ultralytics YOLOv8 Detection**: Accurately detects number plates across challenging angles, vehicle types (cars, cabs, SUVs), and resolutions.
- **Advanced OpenCV Preprocessing**:
  - **Dynamic Margin Trimming**: Crops out plate borders (12% top, 10% bottom, 3% sides) to eliminate frame noise and rivet artifacts.
  - **3× Bicubic Upscaling**: Interpolates small or low-resolution crops to provide legible text strokes.
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes localized lighting variations to enhance character contrast.
- **Smart OCR Cleaning & Indian Plate Normalization**:
  - Whitelist enforcement for uppercase alphanumeric characters `[A-Z0-9]`.
  - Automatic removal of `IND` country identifier prefix artifacts.
  - Regular expression pattern matching (`[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}`) to eliminate stray symbols and characters.
- **SQLite Datastore**:
  - Persistent history log with timestamps, confidence scores, original images, and cropped plate assets.
- **Clean & Lightweight Web UI**:
  - Built with Flask and Vanilla HTML/CSS with instant feedback, file validation, and history inspection.
- **Automated Model Fetching**:
  - Built-in Hugging Face Hub integration that automatically downloads the YOLOv8 weights on first launch.

---

## System Architecture

```mermaid
flowchart TD
    A[Vehicle Image Upload] --> B[Flask Web Application]
    B --> C[YOLOv8 Detection Model]
    C -->|Bounding Box Coordinates| D[Plate Image Cropping]
    
    subgraph Preprocessing Pipeline ["OpenCV Enhancement Pipeline"]
        D --> E[Grayscale Conversion]
        E --> F[Border Noise Trimming]
        F --> G[3x Bicubic Upscaling]
        G --> H[CLAHE Adaptive Contrast]
    end
    
    H --> I[EasyOCR Engine]
    I --> J[Post-Processing & Regex Normalization]
    J -->|DL7CQ1939 / Confidence| K[SQLite Database Logging]
    K --> L[Render Annotated Image & History Table]
```

---

## Project Structure

```text
number-plate-detection/
├── app.py                      # Core Flask web application & OCR inference logic
├── download_model.py           # Automated model weight downloader from Hugging Face
├── requirements.txt            # Python package dependencies
├── .env                        # Local environment configuration
├── .gitignore                  # Git ignore rules for environments, cache & DB
├── plates.db                   # SQLite database for storing detection history
│
├── assets/                     # Documentation media & screenshots
│   ├── detection_preview.png   # Full detection view screenshot
│   ├── history_preview.png     # History table screenshot
│   └── plates/                 # Sample cropped plate showcases
│
├── models/
│   └── best.pt                 # YOLOv8 fine-tuned number plate weights (~6.2 MB)
│
├── templates/
│   └── index.html              # Responsive web UI template
│
├── uploads/                    # Temporary storage for uploaded vehicle images
└── detected_plates/            # Saved plate crops generated during inference
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mohdajnas/AI-number-plate-detection.git
cd AI-number-plate-detection
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
You can either run the download script manually or let `app.py` fetch it automatically on initial startup:
```bash
python download_model.py
```

### 5. Launch the Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## How the OCR Enhancement Works

In `app.py`, raw crops often have low resolution, poor lighting, or border artifacts that confuse standard OCR models. The custom `read_plate()` pipeline addresses this in four distinct steps:

```python
# 1. Grayscale Conversion
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# 2. Border Noise Removal (eliminates plate borders & screws)
h, w = gray.shape
gray = gray[int(h * 0.12):int(h * 0.90), int(w * 0.03):int(w * 0.97)]

# 3. 3x Bicubic Upscaling
gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

# 5. EasyOCR with Alphanumeric Allowlist
results = reader.readtext(gray, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# 6. Format Normalization (strips 'IND' and extracts standard pattern)
match = re.search(r"([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4})", text)
```

---

## Database Schema

The SQLite database (`plates.db`) manages detections using the following schema:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    image_path TEXT NOT NULL,
    plate_image_path TEXT NOT NULL,
    confidence REAL NOT NULL,
    detected_at TEXT NOT NULL
);
```

---

## 🛠️ Tech Stack & Libraries

- **Framework**: [Flask](https://flask.palletsprojects.com/)
- **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (pretrained checkpoint: `Koushim/yolov8-license-plate-detection`)
- **Computer Vision**: [OpenCV (`opencv-python`)](https://opencv.org/)
- **Optical Character Recognition**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Model Hub**: [Hugging Face Hub](https://huggingface.co/)
- **Database**: [SQLite3](https://sqlite.org/)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
