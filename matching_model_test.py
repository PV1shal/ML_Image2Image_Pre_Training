import os
import torch
from PIL import Image
from torchvision import transforms
from II_Implementation import CLIP
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIP(
    embed_dim=512,
    image_resolution=128,
    vision_layers=12,
    vision_width=512,
    vision_patch_size=16,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
).to(device)
model.load_state_dict(torch.load('clip_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

sketch_image_path = r"C:\UNI\CS6140\Project\Dataset\danbooru-sketch-pair-128x\color\sketch\0000\2000.png"
sketch_image = Image.open(sketch_image_path)
sketch_image_tensor = transform(sketch_image).unsqueeze(0).to(device)

src_dir = r"C:\UNI\CS6140\Project\Dataset\Test"

similar_source_images = []
similarities = []
for filename in os.listdir(src_dir):
    src_image_path = os.path.join(src_dir, filename)
    if os.path.isfile(src_image_path):
        src_image = Image.open(src_image_path)
        src_image_tensor = transform(src_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(sketch_image_tensor, src_image_tensor)

        similarity_prob = torch.softmax(logits_per_image, dim=-1).item()
        similarities.append(similarity_prob)
        similar_source_images.append(src_image_path)

print(similarities)
similar_images_sorted = [x for _, x in sorted(zip(similarities, similar_source_images), reverse=True)]
num_top_images = 5
fig, axes = plt.subplots(1, num_top_images+1, figsize=(20, 4))

axes[0].imshow(sketch_image)
axes[0].set_title('Sketch')

for i, img_path in enumerate(similar_images_sorted[:num_top_images]):
    img = Image.open(img_path)
    axes[i+1].imshow(img)
    axes[i+1].set_title(f'Similar Image {i+1}')

plt.tight_layout()
plt.show()