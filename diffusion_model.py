import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from II_Implementation import CLIP
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, image_size, num_steps, clip_model, device):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.num_steps = num_steps
        self.clip_model = clip_model.to(device)
        self.device = device
        
        # Define a diffusion process
        self.diffusion_process = nn.ModuleList([
            self.make_step() for _ in range(num_steps)
        ])
        
        # Define an initial noise sampler
        self.initial_noise_sampler = torch.distributions.normal.Normal(0, 1)
        
    def make_step(self):
        return nn.Sequential(
            nn.Conv2d(513, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 513, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, noise, similarity_scores):
        for t in range(self.num_steps):
            # Concatenate noise and similarity score
            input_tensor = torch.cat((noise, similarity_scores), dim=1)
            noise = self.diffusion_process[t](input_tensor)
            noise = noise / ((t + 1) / self.num_steps)**0.5  # Scale the noise
            if t < self.num_steps - 1:
                noise = torch.randn_like(noise)  # Add noise for next step
        return noise
    
    def generate_samples(self, sketch_image, src_images):
        with torch.no_grad():
            sketch_features = self.clip_model.encode_image(sketch_image).unsqueeze(0).to(self.device)
            similarity_scores = []
            for src_image in src_images:
                src_features = self.clip_model.encode_image(src_image).unsqueeze(0).to(self.device)
                similarity_score = self.clip_model.compute_clip_similarity(sketch_features, src_features)
                similarity_scores.append(similarity_score)
            similarity_scores = torch.tensor(similarity_scores).to(self.device)
            
            noise = self.initial_noise_sampler.sample((1, 3, self.image_size, self.image_size)).to(self.device)
            samples = self.forward(noise, similarity_scores)
        return samples

# Define the training procedure
def train_diffusion_model(diffusion_model, dataloader, optimizer, num_epochs, device):
    diffusion_model.train()
    for epoch in range(num_epochs):
        for sketch_image, src_images in dataloader:
            sketch_image = sketch_image.to(device)
            src_images = [img.to(device) for img in src_images]
            
            # Compute similarity scores using the CLIP model
            sketch_features = diffusion_model.clip_model.encode_image(sketch_image)
            similarity_scores = []
            for src_image in src_images:
                src_features = diffusion_model.clip_model.encode_image(src_image)
                similarity_score = diffusion_model.clip_model.compute_clip_similarity(sketch_features, src_features)
                similarity_scores.append(similarity_score)
            similarity_scores = torch.tensor(similarity_scores).to(device)
            
            # Generate samples
            samples = diffusion_model.generate_samples(sketch_image, src_images)
            
            # Compute loss
            loss = compute_loss(samples, similarity_scores)  # Define your loss function
            
            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Define your loss function
def compute_loss(samples, similarity_scores):
    # Define your loss calculation here
    return torch.tensor(0.0)  # Placeholder, replace with your loss calculation

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the CLIP model
    clip_model_checkpoint_path = 'resnet_test_color.pth'
    clip_model = CLIP(
        embed_dim=512,
        image_resolution=128,
        vision_layers=12,
        vision_width=512,
        vision_patch_size=16,
        use_modified_resnet=True
    ).to(device)
    clip_model.load_state_dict(torch.load(clip_model_checkpoint_path, map_location=device))
    clip_model.eval()
    
    # Instantiate the diffusion model
    image_size = 128
    num_steps = 1000
    diffusion_model = DiffusionModel(image_size, num_steps, clip_model, device)
    
    # Load your dataset and create a DataLoader
    # Replace the following lines with your actual data loading code
    sketch_image_path = r"C:\UNI\CS6140\Project\Dataset\danbooru-sketch-pair-128x\color\sketch\0000\8000.png"
    sketch_image = Image.open(sketch_image_path).convert('RGB')
    
    src_dir = r"C:\UNI\CS6140\Project\Dataset\danbooru-sketch-pair-128x\color\src\0000"
    src_images = [Image.open(os.path.join(src_dir, filename)).convert('RGB') for filename in os.listdir(src_dir)]
    
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(
        root_dir='C:\\UNI\\CS6140\\Project\\Dataset\\danbooru-sketch-pair-128x\\color\\',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer and training parameters
    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.001)
    num_epochs = 10
    
    # Train the diffusion model
    train_diffusion_model(diffusion_model, dataloader, optimizer, num_epochs, device)
