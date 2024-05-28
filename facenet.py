# facenet.py
import numpy as np
from keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN

# Load FaceNet model
model = load_model('facenet_keras.h5')

# Function to get face embedding
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Function to detect faces
def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face
