import sys

import tensorflow as tf
from cv2 import CascadeClassifier, error

model_from_json = tf.keras.models.model_from_json
# from tensorflow.keras.models import model_from_json

from extract_embeddings import ExtractEmbeddings


def loading_models():
    try:
        embedding_obj = ExtractEmbeddings(model_path="models/mobilenetv2_model.keras")
        embedding_model = embedding_obj.load_model()
        face_cascade = CascadeClassifier("models/haarcascade_frontalface_default.xml")

        with open(
            "antispoofing_models/finalyearproject_antispoofing_model_mobilenet.json",
            "r",
        ) as json_file:
            loaded_model_json = json_file.read()
        liveness_model = model_from_json(loaded_model_json)
        # load weights into new model
        liveness_model.load_weights(
            "antispoofing_models/finalyearproject_antispoofing_model_74-0.986316.h5"
        )
        print("Liveness Model loaded successfully from disk")

    except error:
        print("Error: Provide correct path for face detection model.")
        sys.exit(1)
    except Exception as e:
        print(f"{str(e)}")
        sys.exit(1)
    return embedding_obj, embedding_model, face_cascade, liveness_model
