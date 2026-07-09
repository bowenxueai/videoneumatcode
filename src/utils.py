import tempfile, base64, os
import numpy as np
import imageio
from IPython.display import HTML
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import cv2
import tempfile, base64, os
import numpy as np
import imageio
from IPython.display import HTML
import mitsuba as mi
import cv2
import shutil
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.lines import Line2D
import drjit as dr
from copy import deepcopy
from accelerate.logging import get_logger
import torch
from PIL import Image
logger = get_logger(__name__)


def unpack_video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV default) to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames of shape {frames[0].shape}")

    return frames


def plot_loss(loss_values, steps, resolution=(800, 600), ylabel='Loss', xlabel = 'Steps', legend = 'Loss', color = 'blue', font_size = None, linewidth = 2, y_log_scale = True):
    """
    Plots the training loss and returns the plot as a PIL image with the desired resolution.

    Parameters:
        loss_values (list or numpy array): List of training loss values.
        steps (list or numpy array): List of training steps.
        resolution (tuple): Desired resolution of the output image (width, height).

    Returns:
        PIL.Image.Image: The plot as a PIL image.
    """
    # # Validate input lengths
    # if len(loss_values) != len(steps):
    #     raise ValueError(f"The length of loss_values: {len(loss_values)} and steps: {len(steps)} must be the same.")

    # Create the plot
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)  # Width and height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
    if not isinstance(loss_values, list):
        loss_values = [loss_values]
        legend = [legend]
        color = [color]
    

    for i, _ in enumerate(loss_values):
        ax.plot(steps, loss_values[i], label=legend[i], color=color[i], linewidth=linewidth, marker='o', markersize=1)

    # Adding labels and title
    if xlabel:
        ax.set_xlabel('Steps')
    if ylabel:
        ax.set_ylabel(ylabel)
    if y_log_scale:
        ax.set_yscale('log')

    custom_legend_lines = [
        Line2D([0], [0], color=color[_i], linewidth=2, label=legend[_i])
        for _i, _ in enumerate(loss_values)
    ]
    ax.legend(fontsize = font_size, handles=custom_legend_lines)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi = dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)

    # Use PIL to open the image and resize it to the desired resolution
    pil_image = Image.open(buf).convert('RGB')

    # Close the BytesIO buffer
    buf.close()

    # Close the matplotlib figure
    plt.close(fig)

    return pil_image


def ensure_RGB_dims(img, let_1_be_1 = False):
    '''
        Make sure the dimensions are [H,W,3 or 4]

        inputs: 
            img: numpy array
        
        output: 
            Img: numpy array with corrected dimensions

    ''' 

    if len(img.shape) < 2:
        raise Exception("a 1d array passed. image required")
    if len(img.shape) == 2:
        img = img[..., None]
    if len(img.shape) == 3:
        if img.shape[2] == 1 and not let_1_be_1:
            img = img.repeat(3, axis = -1)
        elif img.shape[2] == 2:
            img = np.concatenate([img, np.zeros_like(img)[..., 0:1]],  axis = 2)
    return img


def to_pil(image, resize = None):
    if isinstance(image, torch.Tensor):
        image = image.float().cpu().numpy()

    elif isinstance(image, mi.TensorXf):
        image = torch.tensor(image).cpu().numpy()

    elif isinstance(image, type(Image)):
        return image

    if image.max() > 1 or image.min() < 0.0:
        print(f'warning: image has invalid values in range [{image.min()} - {image.max()}]')
    
    image = (image * 255).astype(np.uint8)

    assert image.ndim == 3
    color_channel = np.argmin(image.shape)
    if color_channel == 0:
        image = image.transpose(1, 2, color_channel)
    image = ensure_RGB_dims(image)

    image = Image.fromarray(image)

    if resize:
        image = image.resize(resize)
    return image





def _to_uint8_rgb_for(frame: mi.TensorXf) -> np.ndarray:
    """Convert Mitsuba TensorXf → uint8 RGB numpy array."""
    if isinstance(frame, np.ndarray) or isinstance(frame, np.array):
        return frame

    bmp = mi.Bitmap(frame)
    bmp_u8 = bmp.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True
    )
    return np.array(bmp_u8, copy=False)

def save_video(frames, fps=24, bitrate=None, resize=None, spp = 32, save_path = None):
    """
    Render a list of Mitsuba mi.TensorXf frames as an inline MP4 in Jupyter.
    """
    # Create a temp file (deleted after embedding)


    if len(frames) == 1:
        rgb = _to_uint8_rgb_for(frames[0])
        Image.fromarray(rgb).save(save_path + '.png')
        return

    save_path += '.mp4'
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        temp_path = tmpfile.name

    # Write the video to disk via FFmpeg
    writer = imageio.get_writer(
        temp_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        bitrate=bitrate or "8M"
    )

    for f in frames:
        rgb = _to_uint8_rgb_for(f)
        if resize is not None:
            rgb = cv2.resize(rgb, resize[::-1])
        writer.append_data(rgb)
    writer.close()
    if save_path is not None:
        shutil.copy(temp_path, save_path)

    # Read file into memory and embed it as base64
    with open(temp_path, "rb") as f:
        video_bytes = f.read()
    os.remove(temp_path)

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    return HTML(f"""
    <video controls autoplay loop style="max-width: 100%;">
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """)



def load_scene_state(state_dict, params, transfer_area_light_to_point_light):

    if transfer_area_light_to_point_light and 'area_light_sphere.emitter.radiance.value' in state_dict.keys():
        state_dict = deepcopy(state_dict)

        area_light_radius = np.asarray(state_dict['area_light_sphere.to_world'])[0,0,0]

        state_dict['light.intensity.value'] = state_dict['area_light_sphere.emitter.radiance.value'] * (area_light_radius**2) * np.pi   # We need to multiply the area light radius to the intensity, since the area light energy is W/unit steradian/unit area
        state_dict['light.position'] = np.asarray(state_dict['area_light_sphere.to_world'])[:3, -1, 0]
        del state_dict['area_light_sphere.emitter.radiance.value']
        del state_dict['area_light_sphere.to_world']


    
    for key, val in state_dict.items():       
        if val is None or key not in params:
            continue
        elif 'to_world' in key:
            params[key] = mi.Transform4f(np.asarray(val)[...,0])
        else:
            params[key] = val
    params.update()




def vec_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=(dr.shape(vec)[1], dr.shape(vec)[0]))




def colormap_diverging(image, vmin=None, vmax=None):
    """
    Colormap an image with negative and positive values.
    - Negative values: Red
    - Zero: Light gray
    - Positive values: Blue
    
    Parameters:
    -----------
    image : np.ndarray
        Input image with negative and positive values
    vmin : float, optional
        Minimum value for normalization. If None, uses image minimum
    vmax : float, optional
        Maximum value for normalization. If None, uses image maximum
    
    Returns:
    --------
    rgb_image : np.ndarray
        RGB image with shape (*image.shape, 3), values in [0, 1]
    """
    # Handle vmin and vmax
    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()
    
    # Ensure symmetric range around zero for better visualization
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
    
    # Normalize to [-1, 1]
    image_norm = np.clip((image - vmin) / (vmax - vmin) * 2 - 1, -1, 1)
    
    # Create RGB image
    rgb_image = np.zeros((*image.shape, 3))
    
    # Light gray color (0.8, 0.8, 0.8) at zero
    gray_value = 0.8
    
    # For negative values: interpolate from red (1, 0, 0) to light gray
    negative_mask = image_norm < 0
    t = np.abs(image_norm[negative_mask])  # 0 at center, 1 at most negative
    rgb_image[negative_mask, 0] = gray_value + (1 - gray_value) * t  # Red channel
    rgb_image[negative_mask, 1] = gray_value * (1 - t)  # Green channel
    rgb_image[negative_mask, 2] = gray_value * (1 - t)  # Blue channel
    
    # For positive values: interpolate from light gray to blue (0, 0, 1)
    positive_mask = image_norm > 0
    t = image_norm[positive_mask]  # 0 at center, 1 at most positive
    rgb_image[positive_mask, 0] = gray_value * (1 - t)  # Red channel
    rgb_image[positive_mask, 1] = gray_value * (1 - t)  # Green channel
    rgb_image[positive_mask, 2] = gray_value + (1 - gray_value) * t  # Blue channel
    
    # For zero values: light gray
    zero_mask = image_norm == 0
    rgb_image[zero_mask] = gray_value
    
    return rgb_image


def remove_module_prefix(state_dict):
    """
    Strips the 'module.' prefix from keys in a state_dict
    saved from a DistributedDataParallel (DDP) model.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module.") :]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict




def save_state(every, accelerator, step, out_dir, metrics, latent_module, safe_tensors):
    if step % every == 0 and step != 0:
        logger.info(f"Process {accelerator.process_index}: Saving state at step {step}", main_process_only=False)
        save_path = os.path.join(out_dir, 'states', f"latest")
        os.makedirs(save_path, exist_ok=True)
        accelerator.save_state(save_path, safe_serialization=safe_tensors)
        torch.save(metrics, os.path.join(save_path, "metrics.pt"))
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(latent_module).state_dict(), f"{os.path.join(out_dir, 'states')}/latent_module.pth")


import gc


def log_cuda_mem(tag: str, accelerator=None):
    """
    Log detailed CUDA memory stats for the current device.

    Prints:
      - device index
      - allocated (live tensors, MB)
      - reserved (PyTorch caching allocator, MB)
      - free / total (raw CUDA, MB)
    """

    device = torch.cuda.current_device()
    # Ensure all pending kernels finish before querying memory
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    free, total = torch.cuda.mem_get_info(device)
    msg = (
        f"****[{tag}] device={device} "
        f"alloc={alloc:.1f} MB reserved={reserved:.1f} MB "
        f"free={free/1024**2:.1f} MB total={total/1024**2:.1f} MB ****"
    )
    logger.info(msg, main_process_only=False, in_order=True)


def is_resumption(out_dir, resume_path):
    path = os.path.join(out_dir, 'states', "latest")
    if os.path.exists(path):
        return 1, path
    elif resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Specified resume path {resume_path} does not exist")
        return 2, resume_path
    else:
        return 0, None



def load_state_if_any(out_dir, accelerator, resume_path=None):
    with accelerator.local_main_process_first():
        resume, path = is_resumption(out_dir, resume_path)
    if resume:
        accelerator.print(f"Resuming {resume} from {['LATEST', 'SPECIFIED'][resume-1]} checkpoint {path}")
        with accelerator.local_main_process_first():
            accelerator.load_state(path)
            metrics = torch.load(os.path.join(path, "metrics.pt"), map_location="cpu")
        return metrics
    else:
        logger.warning(
            f"No saved stated found. Starting a new training run ..."
        )
        return {}