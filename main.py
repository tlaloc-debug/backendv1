import os
from flask import Flask, jsonify, request, abort
from PIL import Image
import requests

from io import BytesIO
from torchvision.models import resnet152, ResNet152_Weights

def load_model():
    # Initialize model with weights
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.eval()
    return model, weights

def download_img(url: str) -> Image:
    # Downloads image to memory.
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


# Initialize app and load global model and weights
app = Flask(__name__)
model, weights = load_model()


@app.route("/hello", methods=["GET"])
def healthcheck():
    return jsonify("Hello!")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Extract url. Return 500 if not included in request.
    url = json.get('url', None)
    if url is None:
        abort(500, "url was not specified in the request.")

    # Get Image from URL
    img = download_img(url)
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    # Get prediction and return
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    category = weights.meta['categories'][class_id]
    app.logger.info({"category":category})
    return jsonify(category=category)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("SERVING_PORT", 8080))