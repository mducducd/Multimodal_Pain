import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torchvision.models as models
from einops import rearrange, repeat

# Load a pre-trained 3D ResNet model from torchvision
class ResNet3DModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet3DModel, self).__init__()
        
        # Use the 3D ResNet-18 model
        self.resnet3d = video_models.r3d_18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

# Custom VGG3D model based on VGG16
class VGG3D(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(VGG3D, self).__init__()
        
        # Load the VGG16 model
        vgg2d = models.vgg16(pretrained=pretrained)
        
        # Modify the features part to handle 3D input
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)), # First Conv3d layer
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # MaxPool3d
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Modify the classifier part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 1, 4096),  # Adjust based on flattened size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # Pass through the modified VGG layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # Fully connected layers
        return x


class ViT3D(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_frames=16, num_classes=3, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(ViT3D, self).__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_per_frame = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dim = dim
        
        # Linear embedding layer for patches
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_dim, dim),
        )

        # Positional encoding that also accounts for the temporal dimension
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches_per_frame * num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)
        
        # Transformer blocks
        self.transformer = nn.Transformer(dim, heads, depth, dim_feedforward=mlp_dim, dropout=0.1)
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, c, t, h, w = x.shape
        
        # Extract patches and flatten them
        x = rearrange(x, 'b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=h//(h//16), p2=w//(w//16))
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.to_patch_embedding(x)
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)
        
        # Apply the transformer
        x = self.transformer(x, x)
        
        # Classification head
        x = x[:, 0]  # CLS token
        x = self.mlp_head(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Input shape: [batch_size=16, channels=3, frames=16, height=224, width=224]
    input_tensor = torch.randn(16, 3, 16, 224, 224)

    # Initialize the 3D ResNet model for 3-class classification
    model = ResNet3DModel(num_classes=3, pretrained=False)

    # Forward pass
    output = model(input_tensor)
    print(output.shape)  # Expected output: [16, 3]