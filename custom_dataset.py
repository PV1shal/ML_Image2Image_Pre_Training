import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        images = []
        sketch_dir = os.path.join(self.root_dir, "sketch")
        src_dir = os.path.join(self.root_dir, "src")

        for subdir in os.listdir(sketch_dir):
            sketch_subdir_path = os.path.join(sketch_dir, subdir)
            src_subdir_path = os.path.join(src_dir, subdir)
            
            if os.path.isdir(sketch_subdir_path) and os.path.isdir(src_subdir_path):
                for filename in os.listdir(sketch_subdir_path):
                    sketch_path = os.path.join(sketch_subdir_path, filename)
                    src_filename = filename.replace("_sketch", "")
                    src_path = os.path.join(src_subdir_path, src_filename)
                    if os.path.isfile(src_path):
                        images.append((sketch_path, src_path))
        return images

    # def load_images(self):
    # images = []
    # sketch_dir = os.path.join(self.root_dir, "sketch")
    # src_dir = os.path.join(self.root_dir, "src")

    # for subdir in os.listdir(sketch_dir):
    #     sketch_subdir_path = os.path.join(sketch_dir, subdir)
    #     src_subdir_path = os.path.join(src_dir, subdir)
        
    #     if os.path.isdir(sketch_subdir_path) and os.path.isdir(src_subdir_path):
    #         for filename in os.listdir(sketch_subdir_path):
    #             sketch_path = os.path.join(sketch_subdir_path, filename)
    #             src_filename = filename.replace("_sketch", "")
    #             src_path = os.path.join(src_subdir_path, src_filename)
    #             if os.path.isfile(src_path):
    #                 try:
    #                     # Attempt to open images
    #                     sketch_image = Image.open(sketch_path).convert('RGB')
    #                     src_image = Image.open(src_path).convert('RGB')
    #                     images.append((sketch_path, src_path))
    #                 except (ValueError, OSError) as e:
    #                     print(f"Error loading image: {e}")
    #                     print(f"Sketch path: {sketch_path}")
    #                     print(f"Src path: {src_path}")
                        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            sketch_path, src_path = self.images[idx]
            sketch_image = Image.open(sketch_path).convert('RGB')
            src_image = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None  # or handle the error in another way
        
        if self.transform:
            sketch_image = self.transform(sketch_image)
            src_image = self.transform(src_image)
        return sketch_image, src_image
