import os
from pickle import loads
from typing import Any, Dict, List, Optional, Tuple

from cv2 import imread
from keras.models import load_model
from numpy import array, ndarray, setdiff1d, unique

rootdir = os.getcwd()


class ExtractEmbeddings:
    """
    The ExtractEmbeddings class provides methods for working with facial embeddings,
    including loading a model, checking a pretrained file, extracting staff details,
    retrieving face pixels, and normalizing pixel values.

    Args:
        model_path (str): The path to the facial embeddings model.

    Attributes:
        model_path (str): The path to the facial embeddings model.
        dataset_dir (str): The directory containing the dataset.

    Methods:
        load_model(): Loads the facial embeddings model.

        check_pretrained_file(embeddings_model: str) -> Tuple[np.ndarray, List[str]]:
            Checks if a pretrained file exists and returns the loaded data and unique names.

        get_staff_details() -> Dict[str, str]:
            Retrieves staff details from the dataset directory.

        get_remaining_names(dictionaries: Dict[str, Any], unique_names: List[str]) -> List[str]:
            Returns a list of names present in dictionaries but not in unique_names.

        get_all_face_pixels(dictionaries: Dict[str, Any]) -> Tuple[List[str], List[str], List[np.ndarray], List[str], List[str]]:
            Retrieves all face pixels, image IDs, paths, arrays, names, and face IDs.

        get_remaining_face_pixels(dictionaries: Dict[str, Any], remaining_names: List[str]) -> Optional[Tuple[List[str], List[str], List[np.ndarray], List[str], List[str]]]:
            Retrieves face pixels for specified remaining names from dictionaries.

        normalize_pixels(imagearrays: List[np.ndarray]) -> np.ndarray:
            Normalizes pixel values of the input image arrays.
    """

    def __init__(self, model_path: str):
        """
        Initializes an instance of the ExtractEmbeddings class.

        Args:
            model_path (str): The path to the facial embeddings model.
        """
        self.model_path = model_path
        self.dataset_dir = os.path.join(rootdir, "dataset")

    def load_model(self) -> Any:
        """
        Loads the facial embeddings model.

        Returns:
            Any: The loaded facial embeddings model.
        """
        return load_model(self.model_path)

    def check_pretrained_file(self, embeddings_model: str) -> Tuple[ndarray, List[str]]:
        """
        Checks if a pretrained file exists and returns the loaded data and unique names.

        Args:
            embeddings_model (str): The path to a pretrained embeddings model.

        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing the loaded data and unique names.
        """
        self.embeddings_model = embeddings_model
        with open(embeddings_model, "rb") as file:
            data = loads(file.read())
        names = array(data["names"])
        unique_names = unique(names).tolist()
        return data, unique_names

    def get_staff_details(self) -> Dict[str, str]:
        """
        Retrieves staff details from the dataset directory.

        Returns:
            Dict[str, str]: A dictionary containing staff names as keys and corresponding IDs as values.
        """
        details = os.listdir(self.dataset_dir)
        staff_details = {item.split("_")[0]: item.split("_")[1] for item in details}
        return staff_details

    def get_remaining_names(
        self, dictionaries: Dict[str, Any], unique_names: List[str]
    ) -> List[str]:
        """
        Returns a list of names present in dictionaries but not in unique_names.

        Args:
            dictionaries (Dict[str, Any]): A dictionary object containing multiple dictionaries.
            unique_names (List[str]): A list of names considered unique.

        Returns:
            List[str]: A list of names present in dictionaries but not in unique_names.
        """
        return setdiff1d(list(dictionaries.keys()), unique_names).tolist()

    def get_all_face_pixels(
        self, dictionaries: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[ndarray], List[str], List[str]]:
        """
        Retrieves all face pixels, image IDs, paths, arrays, names, and face IDs.

        Args:
            dictionaries (Dict[str, Any]): A dictionary containing categories and their corresponding face IDs.

        Returns:
            Tuple[List[str], List[str], List[np.ndarray], List[str], List[str]]:
                A tuple containing lists of image IDs, paths, arrays, names, and face IDs.
        """
        image_ids, image_paths, image_arrays, names, face_ids = [], [], [], [], []

        for category, face_id in dictionaries.items():
            path = os.path.join(self.dataset_dir, f"{category}_{face_id}")
            for img in os.listdir(path):
                img_array = imread(os.path.join(path, img))
                image_paths.append(os.path.join(path, img))
                image_ids.append(img)
                image_arrays.append(img_array)
                names.append(category)
                face_ids.append(face_id)

        return image_ids, image_paths, image_arrays, names, face_ids

    def get_remaining_face_pixels(
        self, dictionaries: Dict[str, Any], remaining_names: List[str]
    ) -> Optional[Tuple[List[str], List[str], List[ndarray], List[str], List[str]]]:
        """
        Retrieves face pixels for specified remaining names from dictionaries.

        Args:
            dictionaries (Dict[str, Any]): A dictionary mapping category names to face IDs.
            remaining_names (List[str]): A list of names representing categories or labels for the images.

        Returns:
            Optional[Tuple[List[str], List[str], List[np.ndarray], List[str], List[str]]]:
                A tuple containing lists of image IDs, paths, arrays, names, and face IDs, or None if remaining_names is empty.
        """
        if not remaining_names:
            return None

        image_ids, image_paths, image_arrays, names, face_ids = [], [], [], [], []

        for category in remaining_names:
            path = os.path.join(
                self.dataset_dir, f"{category}_{dictionaries[category]}"
            )
            for img in os.listdir(path):
                img_array = imread(os.path.join(path, img))
                image_paths.append(os.path.join(path, img))
                image_ids.append(img)
                image_arrays.append(img_array)
                names.append(category)
                face_ids.append(dictionaries[category])

        return image_ids, image_paths, image_arrays, names, face_ids

    def normalize_pixels(self, imagearrays: List[ndarray]) -> ndarray:
        """
        Normalizes pixel values of the input image arrays.

        Args:
            imagearrays (List[np.ndarray]): A list containing the pixel values of multiple images.

        Returns:
            np.ndarray: The normalized pixel values of the input image arrays.
        """
        face_pixels = array(imagearrays).astype("float32")
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        return face_pixels
