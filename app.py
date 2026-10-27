import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import cv2
import easyocr
from flask import Flask, render_template, request
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from werkzeug.utils import secure_filename


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
    """Improve the crop and recognize the text."""
    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Enlarge small plates for better OCR.
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    results = reader.readtext(
        gray,
        detail=0,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = "".join(results).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


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
                "plate_image": plate_filename
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
    app.run(debug=True)