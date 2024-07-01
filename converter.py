import tensorflow as tf
model="models/facenet_keras.h5"
tf.saved_model.save(tf.keras.models.load_model(model), "model_dir/1")
