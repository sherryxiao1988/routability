import torch
import torch.nn as nn
from collections import OrderedDict

PATCH_SIZE = 8
NUM_PATCHES = 1024
# FEATURE_SIZE = 64
# NUM_HEADS = 8
# MLP_SIZE = 128 
NUM_ENCODERS = 5


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class CoordinateEncoder(nn.Module):
    def __init__(self, w=256,h=256):
        super().__init__()
        self.w = w
        self.h = h

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """

        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        input_tensor = input_tensor.cuda()
        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        return out

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
    def __init__(self, dim_in, dim_out, skip_in=0):
        super().__init__()
        # self.process_skip = nn.Sequential(
        #     nn.Conv2d(skip_in, skip_in, kernel_size=3, padding=1),  # You can adjust the number of output channels
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(skip_in)
        # )
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in + skip_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.LeakyReLU(),
                # ResidualBlock(dim_out),
        )

    def forward(self, input, skip=None):
        if skip is not None:
            skip = skip.view(input.shape[0], -1, input.shape[2], input.shape[3])
            # skip = self.process_skip(skip)
            input = torch.cat([input, skip], dim = 1)
        return self.main(input)

class TranformerEncoder(nn.Module):
    def __init__(self, feature_size, num_heads, mlp_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(feature_size, num_heads, mlp_size, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs

class VisionTransformer(nn.Module):
    def __init__(self, FEATURE_SIZE=64, NUM_HEADS=8, MLP_SIZE=128, **kwargs):
        super().__init__()
        # self.coordiEncoder = CoordinateEncoder()
        self.patches = Patches(PATCH_SIZE)
        self.encoded_patches = PatchEncoder(PATCH_SIZE, NUM_PATCHES, FEATURE_SIZE)
        # self.transformer_encoders = nn.Sequential(
        #    *[nn.TransformerEncoderLayer(FEATURE_SIZE, NUM_HEADS, dim_feedforward=MLP_SIZE, batch_first=True) for _ in range(NUM_ENCODERS)],
        # )
        self.transformer_encoders = TranformerEncoder(FEATURE_SIZE, NUM_HEADS, MLP_SIZE, NUM_ENCODERS)

        # now (N, 512, 16, 16)
        self.upconv1 = UpConv(1024, 512)
        self.upconv2 = UpConv(512, 256, 256)
        self.upconv3 = UpConv(256, 128, 64)
        self.upconv4 = UpConv(128, 64, 16)
        self.upconv5 = UpConv(64, 32, 4)

        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.coordiEncoder(x)
        patches = self.patches(x)
        encoded_patches = self.encoded_patches(patches)
        # encoder_output: (N, num_batches, feature_size)
        # [128, 1024, 64]
        encoder_output = self.transformer_encoders(encoded_patches)
        x = encoder_output[-1]
        x = x.view(-1, 1024, 8, 8)
        # x = encoder_output.view(-1, 1024, 8, 8)
        # input (N, 1024, 8, 8)
        x = self.upconv1(x)
        # input (N, 512 + 256, 16, 16)
        x = self.upconv2(x, skip=encoder_output[-2])
        # input (N, 256 + 64, 32, 32)
        x = self.upconv3(x, skip=encoder_output[-3])
        # input (N, 128 + 16, 64, 64)
        x = self.upconv4(x, skip=encoder_output[-4])
        # input (N, 64 + 4, 128, 128)
        x = self.upconv5(x, skip=encoder_output[-5])
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            pass
            # generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

