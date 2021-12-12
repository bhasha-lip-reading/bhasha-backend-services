import os
import argparse
from flask import Flask, request, jsonify, redirect
from werkzeug.utils import secure_filename
from constants import ALLOWED_EXTENSIONS, ASSET_DIR, UPLOAD_DIR
from predictor import predict
from time import time
from collector import collect
import errorhandlers
from appconfig import app

parser = argparse.ArgumentParser()
parser.add_argument('--host')
parser.add_argument('--port')


DEBUG = True


def logger(message, value):
    if DEBUG == True:
        print('#DEBUG: {}: {}'.format(message, value))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload(dir, path):
    logger("Access route", request.access_route)
    logger("Files", request.files)

    if 'video' not in request.files:
        logger('FileNotFoundError', 'video file not exists.')
        return redirect(request.url)

    file = request.files['video']
    logger("video filename", file.filename)

    if file.filename == '':
        logger('FileNotSelectedError', 'no file selected.')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filePath = os.path.join(
            dir, "{}.mp4".format(path))

        logger("File saved to", filePath)

        file.save(filePath)

    return filePath


@app.route("/api/upload", methods=['GET', 'POST'])
def collectData():
    logger('Request url', "/api/upload")
    if request.method == "POST":
        label = request.form.get('label')
        filePath = upload(UPLOAD_DIR, path='{}-{}'.format(label, int(time())))
        collect(filePath)

    return jsonify({
        "url": "/api/upload",
        "method": "GET or POST",
        "code": "200",
        "message": "successfully uploaded"
    })


@app.route("/api/prediction", methods=['GET', 'POST'])
def readLip():
    logger('Request url', "/api/prediction")
    if request.method == "POST":
        # DIR=UPLOAD_DIR, change path="input"
        filePath = upload(ASSET_DIR, path="input")
        return jsonify(
            predict(filePath)
        )

    return jsonify({
        "url": "api/readLip",
        "method": "GET",
        "status": "200",
    })


if __name__ == '__main__':
    args = parser.parse_args()

    if args.host == None:
        args.host = '192.168.0.101'

    if args.port == None:
        args.port = 5000

    app.run(host=args.host, port=args.port)
