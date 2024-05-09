import numpy as np
import torch
from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_coords(dims):
    """
    Generate 2D or 3D coordinate volumes based on specified dimensions.

    This function creates coordinate volumes for either 2D or 3D space, depending on the length of the `dims` tuple provided.
    Each coordinate grid is scaled according to the inverse of the maximum dimension specified in `dims`, ensuring
    that coordinate values are normalized between 0 and 1. For a 2D space, coordinate volumes in the Z-X plane are generated.
    For a 3D space, coordinate volumes in the Z-Y-X space are created.

    Parameters
    ----------
    dims : tuple of int
        A tuple specifying the dimensions (Z, Y, X) or (Z, X) for which to generate coordinate volumes. The length of the 
        tuple determines whether 2D (length 2) or 3D (length 3) coordinate volumes are generated.

    Returns
    -------
    torch.Tensor
        A tensor of shape (Z*X, 2) for 2D input dimensions or (Z*Y*X, 3) for 3D input dimensions, where Z, Y, and X
        correspond to the specified dimensions in `dims`. The tensor contains floating point values for the coordinates,
        normalized based on the maximum dimension size specified in `dims`.
    """
    
    # Calculate pixel size based on the maximum dimension of 'dims'
    pixel_size = 1 / np.max(dims)
    
    # Generate coordinate arrays
    z_ = np.arange(dims[0]) * pixel_size
    x_ = np.arange(dims[1]) * pixel_size
    
    if len(dims) == 3:
        # Handle 3D data
        y_ = np.arange(dims[2]) * pixel_size
        iz, ix, iy = np.meshgrid(z_, x_, y_, indexing='ij')
        Z = torch.tensor(iz, dtype=torch.float32)
        X = torch.tensor(ix, dtype=torch.float32)
        Y = torch.tensor(iy, dtype=torch.float32)
        coords =  torch.stack([Z.ravel(), Y.ravel(), X.ravel()], dim=1)
        
    else:
        # Handle 2D data (dims == 2)
        iz, ix = np.meshgrid(z_, x_, indexing='ij')
        Z = torch.tensor(iz, dtype=torch.float32)
        X = torch.tensor(ix, dtype=torch.float32)
        coords =  torch.stack([Z.ravel(), X.ravel()], dim=1)

    return coords


def windows(dataset, window_size=64, overlap=32):
    """
    Extract 2D windows or 3D subvolumes of a specified size with overlap from a dataset.

    This function slices a dataset into smaller windows, based on the provided `window_size` and `overlap`.
    It supports datasets with either 3D (C, H, W) or 4D (C, D, H, W) dimensions, where C represents channels,
    D depth (for 3D datasets), H height, and W width. The function slices the dataset along the spatial dimensions (D, H, W)
    or (H, W), depending on the dimensionality of the input data.

    Parameters
    ----------
    dataset : np.ndarray or torch.Tensor
        The input dataset to be windowed. It can be a 3D or 4D array/tensor.
    window_size : int, optional
        The size of each window to be extracted. Default is 64.
    overlap : int, optional
        The overlap size between consecutive windows. Default is 32.

    Returns
    -------
    torch.Tensor
        A tensor containing the extracted windows. The windows are stacked along a new dimension at the beginning
        of the return tensor. For 3D data, the shape of the tensor will be (N, C, window_size, window_size, window_size),
        and for 2D data, the shape will be (N, C, window_size, window_size), where N is the total number of windows.
    """

    step = window_size - overlap
    windows = []
    dims = dataset.shape
  
    if dataset.dim() == 4:  # If the dataset is 3D (C, D, H, W)
        C, D, H, W = dataset.shape
        for d in range(0, D - window_size + 1, step):
            for y in range(0, H - window_size + 1, step):
                for x in range(0, W - window_size + 1, step):
                    window = dataset[:, d:d+window_size, y:y+window_size, x:x+window_size]
                    windows.append(window)
                    
    elif dataset.dim() == 3:  # If the dataset is 2D (C, H, W)
        C, H, W = dataset.shape
        for y in range(0, H - window_size + 1, step):
            for x in range(0, W - window_size + 1, step):
                window = dataset[:, y:y+window_size, x:x+window_size]
                windows.append(window)
    else:
        raise ValueError("Unsupported dataset dimensionality. Expected 3 or 4 dimensions, got {}.".format(dataset.dim()))

    return torch.stack(windows)


def windows_rand(dataset, window_size, N):
    """
    Extract N random 2D windows or 3D subvolumes of a specified size from a dataset.

    Given a dataset, this function extracts `N` random windows from it, where each window has a spatial size
    defined by `window_size`. The function supports 4D datasets (with dimensions C, D, H, W representing channels,
    depth, height, and width, respectively) and 3D datasets (with dimensions C, H, W).

    Parameters
    ----------
    dataset : torch.Tensor
        The dataset from which windows will be extracted. It can be a 4D tensor for volumetric data or a 3D tensor for image data.
    window_size : int
        The size of each square window to be extracted. This is applied to height and width for 3D data and to depth, height,
        and width for 4D data.
    N : int
        The number of windows to extract from the dataset.

    Returns
    -------
    torch.Tensor
        A tensor containing the stacked windows extracted from the dataset. The tensor will have a shape of
        (N, C, window_size, window_size) for 3D data, and (N, C, window_size, window_size, window_size) for 4D data.
    """

    windows = []
    if dataset.dim() == 4:  # If the dataset is 4D (C, D, H, W)
        C, D, H, W = dataset.shape
        for _ in range(N):
            od = torch.randint(0, D - window_size, (1,)).item()
            oh = torch.randint(0, H - window_size, (1,)).item()
            ow = torch.randint(0, W - window_size, (1,)).item()
            window = dataset[:, od:od + window_size, oh:oh+ window_size, ow:ow + window_size]
            windows.append(window)

    elif dataset.dim() == 3:  # If the dataset is 3D (C, H, W)
        C, H, W = dataset.shape
        for _ in range(N):
            oh = torch.randint(0, H - window_size, (1,)).item()
            ow = torch.randint(0, W - window_size, (1,)).item()
            window = dataset[:, oh:oh+ window_size, ow:ow + window_size]
            windows.append(window)

    else:
        raise ValueError("Unsupported dataset dimensionality. Expected 3 or 4 dimensions, got {}.".format(dataset.dim()))

    return torch.stack(windows)

class WinDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
        
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample = self.windows[idx]
        return (torch.hstack([s.ravel().unsqueeze(1) for s in sample[:-2]]),
                sample[-2].ravel(), sample[-1].ravel())
