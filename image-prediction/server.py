from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np

import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

# 独自のモデルを定義した場合
#PATH = "####" 
#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()

app = Flask(__name__)
imagenet_class_index = json.load(open('./static/json/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.files['image']:
        file = request.files["image"]
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        print(class_name)
        return render_template('./result.html', title='prediction', class_name=class_name)


if __name__ == '__main__':
    app.run(host="localhost", port=5001, debug=True)