import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import cv2

import numpy as np
import easyocr
from flask import Flask, render_template, request
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
DETECTED_FOLDER = "detected_plates"
MODEL_PATH = "models/best.pt"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)


# Download the model if it is missing.
if not os.path.exists(MODEL_PATH):
    downloaded = hf_hub_download(
        repo_id="Koushim/yolov8-license-plate-detection",
        filename="best.pt"
    )
    import shutil
    shutil.copy(downloaded, MODEL_PATH)


# Load AI models once when the app starts.
detector = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)


def init_db():
    with sqlite3.connect("plates.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                image_path TEXT NOT NULL,
                plate_image_path TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TEXT NOT NULL
            )
        """)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def read_plate(crop):
    if crop.size == 0:
        return ""

    import cv2
    import re

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Remove some border noise
    h, w = gray.shape
    gray = gray[int(h * 0.12):int(h * 0.90),
                int(w * 0.03):int(w * 0.97)]

    # Enlarge the plate
    gray = cv2.resize(
        gray, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # OCR with a restricted character set
    results = reader.readtext(
        gray,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = "".join(results).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Remove common OCR errors around Indian plate formats
    if text.startswith("IND"):
        text = text[3:]

    # Handle extra leading characters
    match = re.search(
        r"([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4})",
        text
    )

    return match.group(1) if match else text


# def read_plate(crop):
#     """Improve the crop and recognize the text."""
#     if crop.size == 0:
#         return ""

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # Enlarge small plates for better OCR.
#     gray = cv2.resize(
#         gray,
#         None,
#         fx=2,
#         fy=2,
#         interpolation=cv2.INTER_CUBIC
#     )

#     gray = cv2.bilateralFilter(gray, 9, 75, 75)

#     results = reader.readtext(
#         gray,
#         detail=0,
#         paragraph=False,
#         allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     )

#     text = "".join(results).upper()
#     text = re.sub(r"[^A-Z0-9]", "", text)

#     return text



def analyze_plate_color(crop):
    if crop is None or crop.size == 0:
        return {
            "background_color": "Unknown",
            "text_color": "Unknown",
            "plate_type": "Unknown"
        }

    # Ignore the outer border
    h, w = crop.shape[:2]

    if h < 5 or w < 5:
        return {
            "background_color": "Unknown",
            "text_color": "Unknown",
            "plate_type": "Unknown"
        }

    inner = crop[
        int(h * 0.15):int(h * 0.85),
        int(w * 0.08):int(w * 0.92)
    ]

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Estimate background using bright, low-saturation pixels
    white_mask = (
        (saturation < 65) &
        (value > 145)
    )

    yellow_mask = (
        (hue >= 15) &
        (hue <= 40) &
        (saturation > 70) &
        (value > 90)
    )

    green_mask = (
        (hue >= 35) &
        (hue <= 90) &
        (saturation > 55) &
        (value > 50)
    )

    black_mask = value < 75

    total = max(value.size, 1)

    white_ratio = np.count_nonzero(white_mask) / total
    yellow_ratio = np.count_nonzero(yellow_mask) / total
    green_ratio = np.count_nonzero(green_mask) / total
    black_ratio = np.count_nonzero(black_mask) / total

    # Choose the most likely background color
    scores = {
        "White": white_ratio,
        "Yellow": yellow_ratio,
        "Green": green_ratio,
        "Black": black_ratio
    }

    background = max(scores, key=scores.get)

    # Estimate character color
    if background in ("White", "Yellow", "Green"):
        text_color = (
            "Black" if black_ratio > 0.05 else "Unknown"
        )
    elif background == "Black":
        text_color = (
            "Yellow" if yellow_ratio > 0.05 else "Unknown"
        )
    else:
        text_color = "Unknown"

    # Heuristic category, not legal verification
    plate_type = "Unknown"

    if background == "White" and text_color == "Black":
        plate_type = "Likely private vehicle"

    elif background == "Yellow" and text_color == "Black":
        plate_type = "Likely commercial/transport"

    elif background == "Green":
        plate_type = "Possible electric vehicle category"

    elif background == "Black" and text_color == "Yellow":
        plate_type = "Possible special-use category"

    return {
        "background_color": background,
        "text_color": text_color,
        "plate_type": plate_type
    }

def save_record(plate, image_path, plate_path, confidence):
    with sqlite3.connect("plates.db") as conn:
        conn.execute(
            """
            INSERT INTO detections
            (plate_number, image_path, plate_image_path,
             confidence, detected_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                plate,
                image_path,
                plate_path,
                confidence,
                datetime.now(timezone.utc).isoformat()
            )
        )


def process_image(filepath):
    image = cv2.imread(filepath)

    if image is None:
        raise ValueError("The uploaded file is not a valid image.")

    results = detector.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    annotated = image.copy()
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Keep the crop inside image boundaries.
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            plate_text = read_plate(crop)
            plate_analysis = analyze_plate_color(crop)

            confidence = float(box.conf[0].item())

            plate_filename = f"{uuid.uuid4().hex}.jpg"
            plate_path = os.path.join(
                DETECTED_FOLDER, plate_filename
            )

            cv2.imwrite(plate_path, crop)

            # Draw the detection on the original image.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = plate_text or "Text not recognized"

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detections.append({
                "plate": plate_text,
                "confidence": round(confidence, 3),
                "plate_image": plate_filename,
                "background_color": plate_analysis["background_color"],
                "text_color": plate_analysis["text_color"],
                "plate_type": plate_analysis["plate_type"]
            })

            if plate_text:
                save_record(
                    plate_text,
                    filepath,
                    plate_path,
                    confidence
                )

    annotated_filename = f"{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(
        DETECTED_FOLDER,
        annotated_filename
    )

    cv2.imwrite(annotated_path, annotated)

    return annotated_filename, detections


@app.route("/", methods=["GET", "POST"])
def index():
    detections = []
    annotated_image = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image."

        elif not allowed_file(file.filename):
            error = "Upload a JPG, JPEG, PNG, or WEBP image."

        else:
            filename = (
                uuid.uuid4().hex
                + "_"
                + secure_filename(file.filename)
            )

            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                annotated_image, detections = process_image(
                    filepath
                )
            except Exception:
                app.logger.exception("Image processing failed")
                error = (
                    "Image processing failed. "
                    "Check the server logs for details."
                )

    with sqlite3.connect("plates.db") as conn:
        conn.row_factory = sqlite3.Row
        history = conn.execute(
            """
            SELECT plate_number, confidence, detected_at
            FROM detections
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()

    return render_template(
        "index.html",
        detections=detections,
        annotated_image=annotated_image,
        history=history,
        error=error
    )


@app.route("/detected/<path:filename>")
def detected_file(filename):
    from flask import send_from_directory

    return send_from_directory(
        DETECTED_FOLDER,
        filename
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=False, use_reloader=False)