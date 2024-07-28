from torchvision import transforms as T
import torch
import numpy as np 

process = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def preprocess(img):
    """Preprocesses an image for use with a PyTorch model."""
    img = process(img)
    # Directly convert to a PyTorch tensor
    img = torch.from_numpy(np.array(img, dtype=np.float32))
    return img