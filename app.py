from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from model.model_def import CustomModel

# =====================
# FLASK INIT
# =====================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/dental_model_v3.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# LOAD MODEL CHECKPOINT
# =====================
checkpoint = torch.load(MODEL_PATH, map_location=device)

CLASS_NAMES = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

model = CustomModel(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# =====================
# IMAGE TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_value = round(confidence.item() * 100, 2)

    return jsonify({
        "class": predicted_class,
        "confidence": confidence_value
    })


# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    app.run(debug=True)
