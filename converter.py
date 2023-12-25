import keras

# Load the model using TensorFlow's Keras API directly
model = keras.models.load_model("keras_model")

# Save the model in a single .keras file using pickle
keras.models.save_model(model, "keras_model.keras")

keras_model_path = "keras_model.keras"
print(f"Model saved in a single .keras file at: {keras_model_path}")

# print(f"Model saved in a single .keras file at: {keras_model_path}")
