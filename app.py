from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="E7uRkyk7iBtHqTqZEw6C"
)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        detection_type = request.form.get("mode", "disease")  # default to disease
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # Save the uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Set model ID based on detection type
        if detection_type == "corn":
            model_id = "maize-001/2"
        else:
            model_id = "corn-disease-odni2/2"

        # Run inference
        result = client.infer(input_path, model_id=model_id)

        # Open the uploaded image
        image = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        predictions = []
        count = 0

        for pred in result.get("predictions", []):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            label = f"{pred['class']} {round(pred['confidence'] * 100, 1)}%"
            x0, y0 = x - w / 2, y - h / 2
            x1, y1 = x + w / 2, y + h / 2

            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0 + 2, y0 - 20), label, fill="cyan", font=font)

            predictions.append({
                "class": pred["class"],
                "confidence": round(pred["confidence"] * 100, 2)
            })

            count += 1

        # Save the annotated image
        output_path = os.path.join(OUTPUT_FOLDER, file.filename)
        image.save(output_path)
        os.remove(input_path)

        return jsonify({
            "mode": detection_type,
            "count": count if detection_type == "corn" else None,
            "predictions": predictions,
            "image_url": f"/result-image/{file.filename}"
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/result-image/<filename>")
def result_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
