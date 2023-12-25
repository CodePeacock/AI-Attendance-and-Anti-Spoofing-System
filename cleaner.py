import os
import shutil


def clean_pycache(directory="."):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".pyc", ".pyo", "__pycache__")):
                os.remove(os.path.join(root, file))
        for dir in dirs:
            if dir == "__pycache__":
                shutil.rmtree(os.path.join(root, dir))


clean_pycache()
