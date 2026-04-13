from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import stackview as sv
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import cv2
import os

data_dir = os.path.join('..', 'data')

def read_raw_image(filepath:str, dtype = np.uint16, shape = (512, 512)) -> np.array:
    """Reads in a CT-Scan slice from the raw file path

    Args:
        filepath (str): File path of the folder
        dtype (type): Data type of the pixels. Defaults to np.uint16
        shape (tuple, optional): Shape of the final array. Defaults to (512, 512).

    Returns:
        np.array: reads in a numpy array of the `shape` parameter
    """

    data = np.fromfile(filepath, dtype=dtype)
    return data.reshape(shape)

def normalize_img(img:np.array) -> np.array:
    """If the image is in Hounsfield Units, it normalizes the pixel values of the image to be between 0 and 1
    
    Args:
        img (np.array): Image to normalize.

    Returns:
        np.array: Normalized image.
    """

    img = img.astype(np.float32)  # ensure float
    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm

def read_scan(scan: str, folder: str, shape = (512, 512)) -> np.array:
    """Reads in the entire 3D scan with shape (Slice, Width, Height)

    Args:
        scan (str): The specific scan name.
        folder (str): The folder to take that scan from.
        shape (tuple, optional): _description_. Defaults to (512, 512).

    Returns:
        np.array: np.array with shape (N, W, H)
    """
    
    if folder == 'raw':
        dtype = np.uint16
        normalize = True
    else:
        dtype = np.float32
        normalize = False
    
    folder = os.path.join(data_dir, folder, scan)
    files = os.listdir(folder)
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    slices = []

    for f in files:
        if f.endswith(".raw"):
            img = read_raw_image(
                os.path.join(folder, f),
                shape=(shape),
                dtype=dtype
            )
            if normalize:
                img = normalize_img(img)
            slices.append(img)

    volume = np.stack(slices, axis=0)
    return volume


def create_side_by_side(vol_1:np.array, vol_2:np.array) -> np.array:
    """Takes 2 CT-Scan volumes and places them next to each other

    Args:
        vol_1 (np.array): Volme 1
        vol_2 (np.array): Volume 2

    Returns:
        np.array: The combined side by side array
    """
    side_by_side = []

    for orig, den in zip(vol_1, vol_2):
        combined = np.concatenate([orig, den], axis=1)  # horizontal stack
        side_by_side.append(combined)

    side_by_side = np.stack(side_by_side, axis=0)
    return side_by_side


def write_manipulated_to_file(volume: np.array, scan: str, type:str, verbose = False):
    """Write a volume to a file

    Args:
        volume (np.array): The Volume you want to write to the file
        scan (str): The name of the scan
        type (str): The type of scan
        verbose (bool, optional): Whether or not to show a progress bar on the writing. Defaults to False.
    """
    num = volume.shape[0]
    base_folder = os.path.join(data_dir, type, scan)
    os.makedirs(base_folder, exist_ok=True)
    if verbose == True:
        for i in tqdm(range(num)):
            file = os.path.join(base_folder, f'{scan}_{i+1:03d}.raw')
            volume[i,:,:].tofile(file)
    else:
        for i in range(num):
            file = os.path.join(base_folder, f'{scan}_{i+1:03d}.raw')
            volume[i,:,:].tofile(file)