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

class VisionTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.patches = Patches(PATCH_SIZE)
        self.encoded_patches = PatchEncoder(PATCH_SIZE, NUM_PATCHES, FEATURE_SIZE)
        self.transformer_encoders = nn.Sequential(
           *[nn.TransformerEncoderLayer(FEATURE_SIZE, NUM_HEADS, dim_feedforward=MLP_SIZE, batch_first=True) for _ in range(NUM_ENCODERS)],
        )

    def forward(self, x):
        patches = self.patches(x)
        encoded_patches = self.encoded_patches(patches)
        encoder_output = self.transformer_encoders(encoded_patches)
        print("encoder output shape: ", encoder_output.shape)

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

