import os
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
import matplotlib.pyplot as plt
from II_Implementation import CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the CLIP model
model = CLIP(
    embed_dim=512,
    image_resolution=128,
    vision_layers=12,
    vision_width=512,
    vision_patch_size=16,
    use_modified_resnet=True
).to(device)

# Path to your modified model checkpoint
# model_checkpoint_path = 'clip_model.pth'
model_checkpoint_path = 'resnet_test_color.pth'
# model_checkpoint_path = 'clip_model.pth'

# Load the model state dict
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.eval()

# Define the preprocessing transformations
transform = Compose([
    Resize((128, 128)),
    Grayscale(num_output_channels=3),
    ToTensor(),
])

# Path to your sketch image: 8000.png
sketch_image_path = r"C:\UNI\CS6140\Project\Dataset\danbooru-sketch-pair-128x\color\sketch\0000\8000.png"
sketch_image = Image.open(sketch_image_path)

# Preprocess the sketch image
sketch_image_tensor = transform(sketch_image).unsqueeze(0).to(device)

src_dir = r"C:\UNI\CS6140\Project\Dataset\danbooru-sketch-pair-128x\color\src\0000"

similar_source_images = []
similarities = []

# Iterate through source images
for filename in os.listdir(src_dir):
    src_image_path = os.path.join(src_dir, filename)
    if os.path.isfile(src_image_path):
        src_image = Image.open(src_image_path)
        src_image_tensor = transform(src_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits_per_image = model(sketch_image_tensor, src_image_tensor)
            
        # Calculate similarity
        similarity_score = logits_per_image.item()
        similarities.append(similarity_score)   

        similar_source_images.append(src_image_path)

# print(similarities)

# Sort similar images based on similarity score
similar_images_sorted = [x for _, x in sorted(zip(similarities, similar_source_images), reverse=True)]
num_top_images = 5

# Display images
fig, axes = plt.subplots(1, num_top_images+1, figsize=(20, 4))
axes[0].imshow(sketch_image)
axes[0].set_title('Sketch')

for i, img_path in enumerate(similar_images_sorted[:num_top_images]):
    img = Image.open(img_path)
    axes[i+1].imshow(img)
    axes[i+1].set_title(f'similarity: {similarities[i]:.6f}')

plt.tight_layout()
plt.show()
