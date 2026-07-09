import torch, warnings
from typing import Optional, Union
from einops import rearrange
from typing import Optional
from typing_extensions import Literal

from .base_pipeline import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit, sinusoidal_embedding_1d
from .model_manager import ModelManager
from .wan_video_dit import WanModel
from .wan_video_vae import WanVideoVAE
from accelerate.logging import get_logger
logger = get_logger(__name__)

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
        ]
        self.post_units = [
            
        ]
        self.model_fn = model_fn_wan_video
    
            


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
        random_init=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    logger.info(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
                random_init=random_init,
                user_kwargs=model_config.user_kwargs
            )
        
        # Load models
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2


        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        # Print configs
        try:
            logger.info("[Config] model_configs:")
            for idx, mc in enumerate(model_configs):
                if mc is None:
                    continue
                logger.info(
                    f"  - [{idx}] model_id={mc.model_id} origin_file_pattern={mc.origin_file_pattern} path={mc.path} offload_device={mc.offload_device} offload_dtype={mc.offload_dtype} local_model_path={mc.local_model_path}",
                    
                )
            if tokenizer_config is not None:
                logger.info(
                    f"[Config] tokenizer_config: model_id={tokenizer_config.model_id} origin_file_pattern={tokenizer_config.origin_file_pattern} path={tokenizer_config.path}",
                    
                )
            if audio_processor_config is not None:
                logger.info(
                    f"[Config] audio_processor_config: model_id={audio_processor_config.model_id} origin_file_pattern={audio_processor_config.origin_file_pattern} path={audio_processor_config.path}",
                    
                )
        except Exception as e:
            warnings.warn(f"Failed to print configs: {e}")
        # Print model architectures
        # try:
        #     logger.info("[Model] Wan DiT architecture:")
        #     logger.info(pipe.dit)
        #     if pipe.dit2 is not None:
        #         logger.info("[Model] Wan DiT-2 architecture:")
        #         logger.info(pipe.dit2)
        #     if pipe.text_encoder is not None:
        #         logger.info("[Model] Text Encoder architecture:")
        #         logger.info(pipe.text_encoder)
        #     if pipe.vae is not None:
        #         logger.info("[Model] VAE architecture:")
        #         logger.info(pipe.vae)
        # except Exception as e:
        #     warnings.warn(f"Failed to print model architectures: {e}")
        return pipe




class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}

    










def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):

    


    if not hasattr(model_fn_wan_video, "_printed_once"):
        model_fn_wan_video._printed_once = False
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if not model_fn_wan_video._printed_once:
        try:
            logger.info(f"[Model] input latents shape: {tuple(latents.shape)}")
            logger.info(f"[Model] text context shape: {tuple(context.shape)}")
            logger.info(f"[Model] timestep shape: {tuple(timestep.shape)}")
        except Exception as e:
            warnings.warn(f"Shape print failed (inputs): {e}")
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    if not model_fn_wan_video._printed_once:
        try:
            logger.info(f"[Model] after patchify (B,C,F,H,W): {tuple(x.shape)}")
        except Exception as e:
            warnings.warn(f"Shape print failed (patchify): {e}")
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    if not model_fn_wan_video._printed_once:
        try:
            logger.info(f"[Model] token sequence shape: {tuple(x.shape)}")
        except Exception as e:
            warnings.warn(f"Shape print failed (tokens): {e}")
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
            
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
        
    if True:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if not model_fn_wan_video._printed_once:
                try:
                    logger.info(f"[Block {block_id}] output shape: {tuple(x.shape)}")
                except Exception as e:
                    warnings.warn(f"Shape print failed (block {block_id}): {e}")
            

    if not model_fn_wan_video._printed_once:
        try:
            logger.info(f"[Model] pre head output shape: {tuple(x.shape)}")
        except Exception as e:
            warnings.warn(f"Shape print failed (pre head): {e}")
        model_fn_wan_video._printed_once = True



    return x, t, (f, h, w)


