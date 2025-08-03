from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Dict, Any, Union

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Roboflow client
client = InferenceHTTPClient(api_url="https://detect.roboflow.com",
                             api_key="E7uRkyk7iBtHqTqZEw6C")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/feedback')
def feedback():
    return render_template("feedback.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        detection_type: str = request.form.get("mode", "disease")
        file = request.files.get("image")

        if not file or not file.filename:
            return jsonify({"error": "No image uploaded"}), 400

        filename: str = file.filename
        input_path: str = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Choose model
        model_id: str = "maize-001/2" if detection_type == "corn" else "corn-disease-odni2/2"

        # Run inference
        result_raw: Union[Dict[str, Any],
                          List[Dict[str,
                                    Any]]] = client.infer(input_path,
                                                          model_id=model_id)

        if not isinstance(result_raw, dict):
            return jsonify({"error": "Unexpected result format"}), 500

        result: Dict[str, Any] = result_raw
        predictions_data: List[Dict[str, Any]] = result.get("predictions", [])
        if not isinstance(predictions_data, list):
            predictions_data = []

        # Open image
        image = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        predictions: List[Dict[str, Any]] = []
        count = 0

        for pred in predictions_data:
            raw_x = pred.get("x")
            raw_y = pred.get("y")
            raw_w = pred.get("width")
            raw_h = pred.get("height")
            cls = pred.get("class", "")
            conf = pred.get("confidence", 0)

            # ✅ Pyright-safe None checks
            if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
                continue

            x: float = float(raw_x)
            y: float = float(raw_y)
            w: float = float(raw_w)
            h: float = float(raw_h)

            x0, y0 = x - w / 2, y - h / 2
            x1, y1 = x + w / 2, y + h / 2

            label = f"{cls} {round(conf * 100, 1)}%"
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0 + 2, y0 - 20), label, fill="cyan", font=font)

            predictions.append({
                "class": cls,
                "confidence": round(conf * 100, 2)
            })
            count += 1

        # Save output image
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        image.save(output_path)
        os.remove(input_path)

        return jsonify({
            "mode": detection_type,
            "count": count if detection_type == "corn" else None,
            "predictions": predictions,
            "image_url": f"/result-image/{filename}"
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/result-image/<filename>")
def result_image(filename: str):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
