import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomDataset
from II_Implementation import CLIP
import torch.nn.functional as F
from tqdm import tqdm

# Define hyperparameters
batch_size = 8
num_epochs = 1
learning_rate = 0.0001

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(
    root_dir='C:\\UNI\\CS6140\\Project\\Dataset\\danbooru-sketch-pair-128x\\color\\',
    transform=transform
)

print(f"Number of images: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

is_resnet = True

# Instantiate CLIP model
model = CLIP(
    embed_dim=512,
    image_resolution=128,
    vision_layers=12,
    vision_width=512,
    vision_patch_size=16,
    use_modified_resnet=is_resnet
).to(device)

# Define loss function (e.g., contrastive loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity_scores, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(similarity_scores, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - similarity_scores, min=0.0), 2))
        return loss_contrastive

loss_fn = ContrastiveLoss().to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}")
#     model.train()
#     total_loss = 0.0

#     for images1, images2 in tqdm(train_loader):
#         optimizer.zero_grad()
#         logits1 = model.encode_image(images1.to(device))
#         logits2 = model.encode_image(images2.to(device))
#         label = torch.zeros(logits1.size(0), dtype=torch.float).to(device)
#         loss = loss_fn(logits1, logits2, label)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     # Print average loss for the epoch
#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    model.train()
    total_loss = 0.0

    for images1, images2 in tqdm(train_loader):
        optimizer.zero_grad()
        similarity_scores = model(images1.to(device), images2.to(device))
        label = torch.ones(similarity_scores.size(0)).to(device)
        loss = loss_fn(similarity_scores, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if is_resnet:
    torch.save(model.state_dict(), 'resnet_test_color.pth')
else:
    torch.save(model.state_dict(), 'vit.pth')