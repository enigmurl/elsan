import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64 * 4, 4, 1, 0),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 0),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 0),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 4, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 1, 3, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Create instances of the generator and discriminator
generator = Generator().to('cuda')
discriminator = Discriminator().to('cuda')

# Define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx in range(10): 
        batch_size = 128
        real_images = 0.5 * torch.zeros((batch_size, 2, 64, 64)).to('cuda')

        # Train Discriminator
        real_labels = torch.ones(batch_size).to('cuda')
        fake_labels = torch.zeros(batch_size).to('cuda')
        
        real_outputs = torch.flatten(discriminator(real_images))
        d_loss_real = criterion(real_outputs, real_labels)
        
        input_noise = torch.randn(batch_size, 100, 1, 1).to('cuda')
        fake_images = generator(input_noise)
        fake_outputs = torch.flatten(discriminator(fake_images))
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        input_noise = torch.randn(batch_size, 100, 1, 1).to('cuda')
        fake_images = generator(input_noise)
        fake_outputs = torch.flatten(discriminator(fake_images))
        
        g_loss = criterion(fake_outputs, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f} true: {torch.mean(torch.abs(1 - fake_images))}")



# Generate and save an all-white image
input_noise = torch.randn(1, 1, 1, 1)
output_image = generator(input_noise)
output_image = output_image.squeeze().detach().numpy() * 255.0
output_image = output_image.astype('uint8')

print("All-white image saved.")

