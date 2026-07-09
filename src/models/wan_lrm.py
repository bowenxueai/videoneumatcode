from .diffsynth_wan_dit.pipeline import WanVideoPipeline, ModelConfig
import torch
from pathlib import Path
from accelerate.logging import get_logger
logger = get_logger(__name__)

def load_LRM_vae(model_path: str | Path, temporal_upsampling, random_init: bool):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.float32,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(model_path), user_kwargs={"allow_temporal_upsampling": temporal_upsampling}),
        ],
        random_init=random_init,
    )

    return pipe.vae

def load_LRM(model_path: str, load_vae: bool, device, null_text_emb_path: str, dtype: str, random_init: bool):


    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=getattr(torch, dtype.lower()),
        device=device,
        model_configs=[
            ModelConfig(path=model_path + "diffusion_pytorch_model.safetensors"),
            # ModelConfig(path=model_path + "Wan2.1_VAE.pth"),
        ],
        random_init=random_init,
    )

    logger.info(f"Device {torch.cuda.current_device()}: Loaded LRM with dtype: {getattr(torch, dtype.lower())}", main_process_only=False, in_order=True)

    
    null_text_emb = torch.load(null_text_emb_path, map_location=device)
    pipe.cached_null_text_emb = null_text_emb
    return pipe

def eval_lrm(pipe: WanVideoPipeline, input_latents, resolution, num_frames, checkpointing: tuple[bool, bool], timestep: int):

    inputs_posi = {}
    inputs_nega = {}
    
    # CFG-unsensitive parameters
    # inputs_shared = {
    #     # Assume you are using this pipeline for inference,
    #     # please fill in the input parameters.
    #     "input_video": data["video"],
    #     "height": data["video"][0].size[1],
    #     "width": data["video"][0].size[0],
    #     "num_frames": len(data["video"]),
    #     # Please do not modify the following parameters
    #     # unless you clearly know what this will cause.
    #     "cfg_scale": 1,
    #     "tiled": False,
    #     "rand_device": pipe.device,
    #     "use_gradient_checkpointing": False,
    #     "use_gradient_checkpointing_offload": False,
    #     "cfg_merge": False,
    #     "vace_scale": 1,
    #     "max_timestep_boundary": 0,
    #     "min_timestep_boundary": 0,
    # }
    
    # # Pipeline units will automatically process the input parameters.
    # for unit in pipe.units:
    #     inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    
    # inputs =  {**inputs_shared, **inputs_posi}

    inputs = {}

    inputs["context"] = pipe.cached_null_text_emb.to(dtype=pipe.torch_dtype, device=pipe.device)
    inputs["input_latents"] = input_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
    inputs["height"] = resolution
    inputs["width"] = resolution
    inputs["num_frames"] = num_frames



    assert len(checkpointing) == 2 and all(isinstance(x, bool) for x in checkpointing), "checkpointing must be a tuple of two booleans"
    timestep = torch.tensor([timestep]).to(dtype=pipe.torch_dtype, device=pipe.device)
    model_output, t, grid_size = pipe.model_fn(dit = pipe.dit, latents = inputs["input_latents"], context = inputs["context"], timestep=timestep, use_gradient_checkpointing=checkpointing[0], use_gradient_checkpointing_offload=checkpointing[1])
    # model_output: b (f h w) d=1536
    return model_output, t, grid_size # the first frame of the generated video is the only one we need

