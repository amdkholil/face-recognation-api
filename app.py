# app.py
from flask import Flask, request, jsonify
import numpy as np
import cv2
from facenet import get_embedding, extract_face, model

app = Flask(__name__)

def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

@app.route('/verify', methods=['POST'])
def verify():
    file1 = request.files['file1']
    file2 = request.files['file2']

    face1 = extract_face(file1)
    face2 = extract_face(file2)

    if face1 is None or face2 is None:
        return jsonify({"error": "Could not detect face in one or both images"}), 400

    embedding1 = get_embedding(model, face1)
    embedding2 = get_embedding(model, face2)

    distance = calculate_distance(embedding1, embedding2)

    threshold = 1.0  # Threshold for considering two faces as the same
    if distance < threshold:
        return jsonify({"match": True, "distance": distance})
    else:
        return jsonify({"match": False, "distance": distance})

if __name__ == '__main__':
    app.run(debug=True)
