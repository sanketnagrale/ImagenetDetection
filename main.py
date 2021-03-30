import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import io
from torchvision import transforms
import json
from flask import Flask, jsonify, request, render_template, redirect
from PIL import Image

app = Flask(__name__)

model = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)
model.eval()
imagenet_class_index = json.load(open('imagenet_class_index.json'))


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    image = Image.open(io.BytesIO(image))
    return transform(image)


def get_prediction(image):
    tensor = transform_image(image)
    batch = torch.unsqueeze(tensor, 0)
    out = model(batch)
    _, index = torch.max(out, 1)
    predicted_id = str(index.item())
    return imagenet_class_index[predicted_id]


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            rerturn
        img = file.read()
        class_id, class_name = get_prediction(img)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
