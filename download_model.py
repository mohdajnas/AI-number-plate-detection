from huggingface_hub import hf_hub_download
import shutil
import os

os.makedirs("models", exist_ok=True)

model_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

shutil.copy(model_path, "models/best.pt")

print("Model downloaded successfully!")