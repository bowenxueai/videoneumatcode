import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
from .wan_lrm import load_LRM, eval_lrm
from einops import rearrange
from accelerate.logging import get_logger
from .diffsynth_wan_dit.upsamplers import initialize_upsampler
logger = get_logger(__name__)


class LatentTexture(nn.Module):
    def forward(self, material_names, input_images):
        pass



class StandaloneTexture(LatentTexture):
    def __init__(self, material_names, resolution, channels, device, **kwargs):
        super().__init__()
        material_names = sorted(material_names)  # Ensure consistent ordering across ranks
        num_materials = len(material_names)
        self.names_to_index = {name: i for i, name in enumerate(material_names)}
        self.data = torch.nn.Parameter(torch.randn(num_materials, channels, resolution, resolution)*0.1)
        logger.info(f"{kwargs} not used in StandaloneTexture")

    def get_extra_state(self):
        return self.names_to_index

    def set_extra_state(self, state):
        self.names_to_index = state
        # Resize data if number of materials changed (will be overwritten by load_state_dict)
        new_num_mats = len(self.names_to_index)
        if self.data.shape[0] != new_num_mats:
            self.data = torch.nn.Parameter(torch.zeros(new_num_mats, *self.data.shape[1:], device=self.data.device, dtype=self.data.dtype))

    def forward(self, material_names, _, checkpointing: tuple[bool, bool]):
        """
        Assumption on the shape of inputs:
            uv: [B,..., 2]
            material_names: [B]
        """
        B = len(material_names)
        num_mats = len(self.data)
        if num_mats == 1:
            textures = self.data.expand(B, -1, -1, -1) # for efficiency, we can make a view instead of a copy
        else:
            material_ids = torch.tensor([self.names_to_index[mat_name] for mat_name in material_names])
            textures = self.data[material_ids.to(self.data.device)]

        return textures
    
    def blur_latents(self, mat_names, sigma):
        mat_names = set(mat_names)
        mat_id = torch.tensor([self.names_to_index[mat_name] for mat_name in mat_names]).to(self.data.device)
        with torch.no_grad():
            self.data.data[mat_id] = gaussian_blur(self.data[mat_id], kernel_size= sigma)


class NeuMIPStandaloneLatents(LatentTexture):
    def __init__(self, use_offset, **texture_desc):
        super().__init__()
        self.latents = StandaloneTexture(**texture_desc)
        self.offset_latents = StandaloneTexture(**texture_desc) if use_offset else None


    def forward(self, material_names, _, checkpointing: tuple[bool, bool]):
        return self.latents(material_names, _, checkpointing=checkpointing), self.offset_latents(material_names, _, checkpointing=checkpointing) if self.offset_latents is not None else None


    def blur_latents(self, mat_id, sigma):
        self.latents.blur_latents(mat_id, sigma)
        if self.offset_latents is not None:
            self.offset_latents.blur_latents(mat_id, sigma + 2)
    

    def set_trainable(self, trainable: bool):
        self.latents.data.requires_grad_(trainable)
        if self.offset_latents is not None:
            self.offset_latents.data.requires_grad_(trainable)

    def get_parameters(self):
        return list(self.parameters())



class WANLRMTexture(LatentTexture):
    def __init__(self, model_path, load_vae, device, null_text_emb_path, input_resolution, n_input_frames, dtype, channels, upsampler, timestep, random_init, **kwargs):
        super().__init__()
        self.LRM = load_LRM(model_path, load_vae, device, null_text_emb_path, dtype, random_init)
        self.input_resolution = input_resolution
        self.n_input_frames = n_input_frames
        self.channels = channels
        self.timestep = timestep
        logger.info(f"Using WAN at timestep {timestep}")
        wan_lrm_head = self.LRM.dit.head.state_dict().copy()
        del self.LRM.dit.head
        self.upsampler = initialize_upsampler(**upsampler, patch_size=self.LRM.dit.patch_size, latent_channels=channels, factor=8, dim = 1536, out_dim = 16, eps = self.LRM.dit.eps_default, wan_dit_head_weights = wan_lrm_head)
        logger.info(f"Initialized upsampler: {upsampler}, \n{self.upsampler}")
        logger.info(f"{kwargs} not used in WANLRMTexture")
        self.printed_once = False


    def forward(self, material_ids, input_images, checkpointing: tuple[bool, bool]):
        if input_images.ndim ==4:
            input_images = input_images[None]

        lrm_output, t, grid_size = eval_lrm(self.LRM, input_images, self.input_resolution, self.n_input_frames, checkpointing=checkpointing, timestep=self.timestep)
        upsampled_output = self.upsampler(lrm_output, t, grid_size)
        b, c, f, h, w = upsampled_output.shape
        assert c == self.channels * 2, f"Expected {self.channels * 2} channels, got {c}"
        # assert (h == self.input_resolution) and (w == self.input_resolution), f"Expected input resolution {self.input_resolution}, got {h}x{w}"
        if not self.printed_once:
            logger.info(f"LRM WAN output shape: {lrm_output.shape}, Upsampled output shape: {upsampled_output.shape}")
            self.printed_once = True
        upsampled_output = upsampled_output[:, :, 0] # we only need the first frame
        latents, offset_latents = upsampled_output.chunk(2, dim = 1)
        return latents, offset_latents

    def blur_latents(self, mat_id, sigma):
        pass

    def set_trainable(self, trainable: bool):
        self.LRM.freeze_except(['dit'] if trainable else [])
        self.upsampler.train(trainable)
        [p.requires_grad_(trainable) for p in self.upsampler.parameters()]

    def get_parameters(self):
        return list(self.LRM.dit.parameters()) + list(self.upsampler.parameters())


    def eval(self):
        self.LRM.eval()
        self.upsampler.eval()
        
    def train(self, mode: bool = True):
        self.LRM.train(mode)
        self.upsampler.train(mode)


class SimpleLRMTexture(LatentTexture):
    def __init__(self, n_input_frames, **kwargs):
        super().__init__()
        self.LRM = SimpleLRM(embed_dim=512, patch_size=4, in_frames=n_input_frames)
        self.n_input_frames = n_input_frames
        logger.info(f"{kwargs} not used in SimpleLRMTexture")


    def forward(self, material_ids, input_images, checkpointing: tuple[bool, bool]):
        if input_images.ndim ==4:
            input_images = input_images[None]
        input_images = rearrange(input_images, 'b t h w c -> b t c h w')

        lrm_output = self.LRM(input_images)
        latents, offset_latents = lrm_output.chunk(2, dim = 1)
        return latents, offset_latents

    def blur_latents(self, mat_id, sigma):
        pass

    def set_trainable(self, trainable: bool):
        self.LRM.train(trainable)
        [p.requires_grad_(trainable) for p in self.LRM.parameters()]

    def get_parameters(self):
        return self.LRM.parameters()


    def eval(self):
        self.LRM.eval()
        
    def train(self, mode: bool = True):
        self.LRM.train(mode)