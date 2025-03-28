import os
import numpy as np
from PIL import Image
import torch

class ImageLoader:
    def load_images_to_arrays(self, folder_path):
        images = []
        for file in os.listdir(folder_path):
            if file.endswith('jpeg'):
                img_path = os.path.join(folder_path, file)
                IMG = Image.open(img_path).convert('RGB')  # Open the image in RGB format
                IMG = IMG.resize((128, 128))  # Resize to 128x128
                img_array = np.array(IMG)
                images.append(img_array)
        
        images = torch.tensor(np.stack(images), dtype=torch.float32)  # Convert list of images to tensor

        images = images / 255.0  # Normalize the images to [0, 1]

        images = images.permute(0, 3, 1, 2)  # Convert to shape (batch_size, channels, height, width)

        return images

    def flatten_images(self, data):
        # Flatten each image (batch_size, channels, height, width) -> (batch_size, channels*height*width)
        return data.reshape(data.size(0), -1)  # Flatten each image in the batch
