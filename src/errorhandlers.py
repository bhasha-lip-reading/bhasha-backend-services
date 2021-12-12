from appconfig import app
from flask import json, jsonify


@app.errorhandler(404)
def NotFoundError(e):
    return jsonify({
        "code": 404,
        "message": "url not found error",
    })


@app.errorhandler(500)
def ServerError(e):
    return jsonify({
        "code": 500,
        "message": "server error",
    })
