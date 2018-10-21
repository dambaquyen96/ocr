# -*- coding: utf-8 -*-
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, send_from_directory, json, jsonify
from utils import denoise_segment
import io
import numpy as np
import cv2
import os
from uuid import uuid4

UPLOAD_FOLDER = 'web/img/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
template_dir = os.path.abspath('web')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)

@app.route('/web/img/<path:path>')
def load_shards_1(path):
    return send_from_directory('web/img', path)

@app.route('/web/<path:path>')
def load_shards_2(path):
    return send_from_directory('web', path)

@app.route('/web/imgareaselect/<path:path>')
def load_shards_3(path):
    return send_from_directory('web/imgareaselect', path)

@app.route('/web/imgareaselect/css/<path:path>')
def load_shards_4(path):
    return send_from_directory('web/imgareaselect/css', path)

@app.route('/web/imgareaselect/js/<path:path>')
def load_shards_5(path):
    return send_from_directory('web/imgareaselect/js', path)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/api/prepare", methods=["POST"])
def prepare():
    name = str(uuid4())
    file = request.files['file']
    W = int(request.values['total_width'])
    H = int(request.values['total_height'])
    w = int(request.values['width'])
    h = int(request.values['height'])
    x = int(request.values['x'])
    y = int(request.values['y'])

    input_img = decode_img(file, W, H, w, h, x, y)
    cv2.imwrite("web/img/{}-origin.jpg".format(name), input_img)

    n, bboxes, images = denoise_segment.segment_character(input_img)
    result = dict()
    result['segment'] = []
    result['segment_jpg'] = []
    for i in range(n):
        x, y, w, h, _ = np.squeeze(bboxes[i])
        img = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        name_jpg = "web/img/{}-{:02d}.jpg".format(name, i)
        cv2.imwrite(name_jpg, img)
        result['segment_jpg'].append(name_jpg)
        img_tf = img.astype(np.float64)
        img_tf *= 1./255
        result['segment'].append(img_tf.flatten().tolist())

    name_processed = "web/img/{}-processed.jpg".format(name)
    cv2.imwrite(name_processed, input_img)
    result['image'] = name_processed

    return json.dumps(result, ensure_ascii=False)

@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)

def decode_img(file, WIDTH, HEIGHT, w, h, x, y):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)
    scale = max(1.*img.shape[0]/HEIGHT, 1.*img.shape[1]/WIDTH)
    w = int(scale * w)
    h = int(scale * h)
    x = int(scale * x)
    y = int(scale * y)
    res = img[y:y+h, x:x+w, :]
    return res

if __name__ == '__main__':
    app.run()