import torch
import torch.nn as nn

PATCH_SIZE = 8
NUM_PATCHES = 1024
FEATURE_SIZE = 64
NUM_HEADS = 2
MLP_SIZE = 32
NUM_ENCODERS = 4


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channel_size, height, width = images.size()
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # each patch is projected into P^2*C
        # number of patches is HW / P^2
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channel_size)
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, patch_size, num_patches, transformer_feature_size):
        super().__init__()
        self.num_patches = num_patches
        # The Transformer uses constant latent vector size D through all of its layers, 
        # so we flatten the patches and map to D dimensions with a trainable linear projection
        # here from 192 -> 64
        self.projection = nn.Linear(patch_size * patch_size * 3, transformer_feature_size)
        self.position_embedding = nn.Embedding(num_patches, transformer_feature_size)

    def forward(self, patch):
        positions = torch.arange(0, self.num_patches).to(patch.device)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class ResidualBlock(nn.Module):
    def __init__(self, filters, downsample=False, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(filters)
        
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(filters)

        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1,
                                             stride=2, padding=0, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(self.relu1(out))
        
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
        
        out += identity
        out = self.bn2(self.relu2(out))
        
        return out

class UpConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.LeakyReLU(),
                ResidualBlock(dim_out),
        )

    def forward(self, input):
        return self.main(input)

class VisionTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.patches = Patches(PATCH_SIZE)
        self.encoded_patches = PatchEncoder(PATCH_SIZE, NUM_PATCHES, FEATURE_SIZE)
        self.transformer_encoders = nn.Sequential(
           *[nn.TransformerEncoderLayer(FEATURE_SIZE, NUM_HEADS, dim_feedforward=MLP_SIZE, batch_first=True) for _ in range(NUM_ENCODERS)],
        )

        # now (N, 512, 16, 16)
        self.upconv1 = UpConv(1024, 512)
        self.upconv2 = UpConv(512, 256)
        self.upconv3 = UpConv(256, 128)
        self.upconv4 = UpConv(128, 64)
        self.upconv5 = UpConv(64, 32)

        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        patches = self.patches(x)
        encoded_patches = self.encoded_patches(patches)
        # encoder_output: (N, num_batches, feature_size)
        # [128, 1024, 64]
        encoder_output = self.transformer_encoders(encoded_patches)
        x = encoder_output.view(-1, 1024, 8, 8)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        pass
        # if isinstance(pretrained, str):
        #     new_dict = OrderedDict()
        #     weight = torch.load(pretrained, map_location='cpu')['state_dict']
        #     for k in weight.keys():
        #         new_dict[k] = weight[k]
        #     load_state_dict(self, new_dict, strict=strict, logger=None)
        # elif pretrained is None:
        #     generation_init_weights(self)
        # else:
        #     raise TypeError("'pretrained' must be a str or None. "
        #                     f'But received {type(pretrained)}.')

