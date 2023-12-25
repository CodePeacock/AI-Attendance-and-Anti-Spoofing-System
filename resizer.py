from PIL import Image
import os

def resize_images(path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust for other extensions if needed
            img = Image.open(os.path.join(path, filename))
            img = img.resize((160, 160), Image.ANTIALIAS)  # Maintain quality with ANTIALIAS
            img.save(os.path.join(path, filename))

# Example usage:
image_path = "dataset/Mayur Sinalkar_9"  # Replace with your actual path
resize_images(image_path)
