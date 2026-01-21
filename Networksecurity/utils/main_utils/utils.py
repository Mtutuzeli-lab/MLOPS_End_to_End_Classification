"""
Main Utility Functions
Helper functions for saving/loading objects, numpy arrays, and other common operations
"""

import os
import sys
import pickle
import numpy as np
import yaml
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array to file
    
    Args:
        file_path (str): Path where to save the array
        array (np.array): Numpy array to save
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save numpy array
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        
        logging.info(f"Numpy array saved successfully at: {file_path}")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array from file
    
    Args:
        file_path (str): Path to the numpy array file
        
    Returns:
        np.array: Loaded numpy array
    """
    try:
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        
        logging.info(f"Numpy array loaded successfully from: {file_path}")
        return array
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: Any) -> None:
    """
    Save Python object using pickle
    
    Args:
        file_path (str): Path where to save the object
        obj (Any): Python object to save (e.g., trained model, preprocessor)
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load Python object from pickle file
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        Any: Loaded Python object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Write YAML file
    
    Args:
        file_path (str): Path where to save the YAML file
        content (dict): Dictionary content to write
        replace (bool): Whether to replace existing file
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Write YAML file
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        
        logging.info(f"YAML file saved successfully at: {file_path}")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Content of YAML file as dictionary
    """
    try:
        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)
        
        logging.info(f"YAML file read successfully from: {file_path}")
        return content
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
