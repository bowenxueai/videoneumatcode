
from typing import Any
import torch
from src.utils import unpack_video_to_numpy
from PIL import Image
import numpy as np
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from src.common import tonemapping
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
from accelerate.logging import get_logger
from decord import VideoReader, cpu
import gc
from omegaconf import OmegaConf
import h5py

logger = get_logger(__name__)

def spherical_to_cartesian_numpy(spherical: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates (theta, phi) to Cartesian (x,y,z) using NumPy.
    
    Args:
        spherical: numpy array of shape [..., 2] containing (theta, phi)
        
    Returns:
        numpy array of shape [..., 3] containing unit vectors (x, y, z)
    """
    theta, phi = spherical[..., 0], spherical[..., 1]
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def spherical_to_cartesian(spherical: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates (theta, phi) back to Cartesian (x,y,z).
    
    Args:
        spherical: Tensor of shape [..., 2] containing (theta, phi)
        
    Returns:
        Tensor of shape [..., 3] containing unit vectors (x, y, z)
    """
    theta, phi = spherical[..., 0], spherical[..., 1]
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def load_scene_states(scene_state_path, n_frames):
    states = [np.load(f'{scene_state_path}/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]
    return states


def open_video(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise OSError(f"Error opening video {video_path}: {e}")
    opener = {
        "vr": vr,
        "total": len(vr),
        "path": video_path
    }
    return opener

def read_frames(opener, frame_list=None):
    opener = open_video(opener)
    vr = opener["vr"]
    total = opener["total"]

    if frame_list is None:
        return [vr[i].asnumpy().astype(np.float32) / 255.0 for i in range(total)]

    frames = []
    for idx in frame_list:
        if idx < 0 or idx >= total:
            raise IndexError(f"Frame {idx} out of range [0, {total - 1}]")

        arr = vr[idx].asnumpy().astype(np.float32) / 255.0
        frames.append(arr)
    del vr, opener
    gc.collect()  # Force garbage collection to release VideoReader memory

    return frames

from time import time
class SingleMaterial(torch.utils.data.Dataset):
    def __init__(self, material_name, material_path, dataset_n_frames, resolution, tonemapping_type, input_type, frame_list, representative_frame_list, rep_images_use_latent, lazy_load = False, states = None, rep_images_path = None, split = None, traj = None, rep_resolution = None, use_hdf5 = False, hdf5_data_root = None, hdf5_n_slices = 64):
        self.material_name = material_name
        self.input_type = input_type
        self.tonemapping_type = tonemapping_type

        self.resolution = resolution
        self.rep_resolution = rep_resolution if rep_resolution is not None else resolution
        self.dataset_n_frames = dataset_n_frames
        self.path = material_path
        self.lazy_load = lazy_load
        # lazy_load = self.lazy_load
        self.frame_list = frame_list
        self.use_hdf5 = use_hdf5
        self.hdf5_data_root = hdf5_data_root
        self.hdf5_n_slices = hdf5_n_slices
        self.hdf5_path = Path(hdf5_data_root) / material_name if use_hdf5 and hdf5_data_root else None
        self.hdf5_data = None

        # logger.info(f"material: {material_name}, frame_list: {frame_list}")

        self.representative_frame_list = representative_frame_list if representative_frame_list is not None else []  # list of frames that are representative of the material (for LRM)
        self.vae_latent_path = None
        self.rep_images_path = rep_images_path
        self.rep_images = None
        self.rep_images_use_latent = rep_images_use_latent
        if representative_frame_list:
            self.vae_latent_path = Path(self.rep_images_path) / '0 latents'/ f'{sorted(representative_frame_list)}-{self.rep_resolution}' / f'{material_name}.pt'
            self.rep_images = self.load_reps() if not lazy_load else None

        if use_hdf5:
            self.size = hdf5_n_slices  # number of slices
            if not lazy_load:
                self.hdf5_data = self.load_hdf5_all()
        else:
            self.input_images = None if lazy_load else self.load_inputs(frame_list)
            self.input_images = None if lazy_load else self.post_process(self.input_images)

            self.size = len(frame_list)
            self.traj = traj
            self.states = states


        self.split = split

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.use_hdf5:
            return self.load_hdf5_slice(index)
        if self.lazy_load:
            input_images = self.load_inputs([self.frame_list[index]])
            input_images = self.post_process(input_images)
            return input_images[0]
        return self.input_images[index]
    
    def load_hdf5_slice(self, slice_idx):
        """Load a single slice from HDF5 file."""
        start_time = time()
        
        if self.hdf5_data is not None:
            raise Exception()
            uv = self.hdf5_data['uvs'][slice_idx]
            wi_polar = self.hdf5_data['wis'][slice_idx]
            wo_polar = self.hdf5_data['wos'][slice_idx]
            target = self.hdf5_data['targets'][slice_idx]
        else:
            arr = np.load(self.hdf5_path /f'{slice_idx:04d}.npy')
            uv     = arr[..., 0:2].astype(np.float32)
            wi_polar     = arr[..., 2:4].astype(np.float32)   # polar (theta, phi)
            wo_polar     = arr[..., 4:6].astype(np.float32)   # polar (theta, phi)
            target = arr[..., 6:9].astype(np.float32)

        end_time = time()
        # logger.info(f"Time taken to load hdf5 slice {slice_idx}: {end_time - start_time} seconds")
        
        wi = spherical_to_cartesian_numpy(wi_polar)
        wo = spherical_to_cartesian_numpy(wo_polar)
        
        return {
            'uv': torch.from_numpy(uv),
            'wi': torch.from_numpy(wi),
            'wo': torch.from_numpy(wo),
            'target': torch.from_numpy(target).clamp(0.0, 4.0),
        }
    
    def load_hdf5_all(self):
        """Load all slices from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            logger.info(f"Loaded all slices from {self.hdf5_path}, shape of uvs: {f['uvs'][:].shape}")

            return {
                'uvs': f['uvs'][:].astype(np.float32),
                'wis': f['wis'][:].astype(np.float32),
                'wos': f['wos'][:].astype(np.float32),
                'targets': f['targets'][:].astype(np.float32),
            }

    def get_rep_images(self):
        return self.load_reps() if self.lazy_load else self.rep_images

        
    def load_reps(self):
        if self.rep_images_use_latent:
            return torch.load(self.vae_latent_path, map_location='cpu') if self.vae_latent_path is not None else None
        else:
            out =  self.load_inputs(self.representative_frame_list, path = self.rep_images_path) if self.representative_frame_list else None
            return self.post_process(out, resolution = self.rep_resolution) if out is not None else None

    def load_inputs(self, frame_list, path = None):
        if path is None:
            path = str(self.path)
        input_type = self.input_type


        if input_type =='video':
            gts = read_frames(f'{self.path}/btf_sequence.mp4', frame_list)
            
        elif input_type == 'exr':
            gts = [mi.TensorXf(mi.Bitmap(f'{path}/exr/btf_frame__{i:04d}.exr')).numpy() for i in frame_list]
                
        elif input_type == 'png':
            gts = [(np.asarray(Image.open(f'{path}/png/btf_frame__{i:04d}.png'))/255.0) for i in frame_list]
            

        else:
            raise ValueError(f'Invalid input type: {input_type}')
        return gts


    def post_process(self, images, resolution = None):
        if resolution is None:
            resolution = self.resolution
        images = torch.stack([torch.from_numpy(img).float() for img in images])
        images = rearrange(images, 'b h w c -> b c h w')

        if self.input_type == 'exr':
            images = tonemapping(self.tonemapping_type, images)
        if images[-1].shape[0] != resolution: # assuming: b c h w
            images = torch.nn.functional.interpolate(images, size=(resolution, resolution), mode='bilinear')

        images = rearrange(images, 'b c h w -> b h w c')

        return images
    
    


class MultipleMaterials(torch.utils.data.Dataset):
    def __init__(self, materials: dict[str, SingleMaterial], unrolled: bool, groups_by_name: dict[str, list[str]]):
        # names are like "cfg_name***{material_name}"
        self.names, self.datasets = list(materials.keys()), list(materials.values())
        self.names_to_index = {name: i for i, name in enumerate(self.names)}
        self.real_names = list(set([dataset.material_name for dataset in self.datasets]))
        self.n_mats= len(self.datasets)
        self.unrolled = unrolled
        self.use_hdf5 = self.datasets[0].use_hdf5 if self.datasets else False
        
        def build_index_list(datasets):
            mapping = []
            for d_idx, d in enumerate(datasets):
                if d.use_hdf5:
                    mapping.extend([d_idx, i] for i in range(d.hdf5_n_slices))
                else:
                    mapping.extend([d_idx, i] for i in range(len(d)))
            return mapping

        if unrolled:
            self.index_list = build_index_list(self.datasets)

        if groups_by_name is not None:
            self.groups: dict[str, list[int]] = {}
            for group_name, material_names in groups_by_name.items():
                self.groups[group_name] = [self.names_to_index[n] for n in material_names]




    def __len__(self):
        return len(self.index_list) if self.unrolled else self.n_mats


    def __getitem__(self, id):
        if self.unrolled:
            return self.get_item_unrolled(id)
        else:
            return self.get_item_per_material(id)

    def get_item_per_material(self, id):
        mat_index = id

        dataset = self.datasets[mat_index]
        size = len(dataset)
        
        # Load all frames at once to avoid opening video multiple times
        if dataset.lazy_load and dataset.input_type == 'video':
            input_images = dataset.load_inputs(dataset.frame_list)
            images = dataset.post_process(input_images)
        else:
            images = torch.stack([dataset[i] for i in range(size)], dim=0)

        output = {
            'images': images, 
            'mat_ids': torch.tensor([mat_index]), 
            'view_ids': torch.tensor(list(range(size))), 
            "sample_id": torch.tensor([mat_index]), 
            'names': dataset.material_name, 
            'split': dataset.split, 
            'traj': dataset.traj
            }

        rep_images = dataset.get_rep_images()
        if rep_images is not None:
            output['rep_images'] = rep_images

        return output


    def get_item_unrolled(self, id):
        mat_index, slice_idx = self.index_list[id]
        dataset = self.datasets[mat_index]

        if dataset is None:
            raise ValueError(f"Dataset {mat_index} {self.names[mat_index]} is None")

        if dataset.use_hdf5:
            hdf5_data = dataset[slice_idx]
            output = {
                'uv': hdf5_data['uv'],  # (1024, 1024, 2)
                'wi': hdf5_data['wi'],  # (1024, 1024, 3)
                'wo': hdf5_data['wo'],  # (1024, 1024, 3)
                'target': hdf5_data['target'],  # (1024, 1024, 3)
                'mat_ids': torch.tensor([mat_index]),
                'slice_ids': torch.tensor([slice_idx]),
                'sample_id': id,
                'names': dataset.material_name,
                'split': dataset.split,
            }
        else:
            images = dataset[slice_idx]
            output = {
                'images': images, 
                'mat_ids': mat_index, 
                'view_ids': slice_idx, 
                "sample_id": id, 
                'names': dataset.material_name,
                'split': dataset.split, 
                'traj': dataset.traj, 
            }

        rep_images = self.datasets[mat_index].get_rep_images()
        if rep_images is not None:
            output['rep_images'] = rep_images
        return output

    def get_by_name_and_view_id(self, name, view_id):
        assert isinstance(name, str) and view_id is not None
        each_size = self.datasets[0].size
        mat_index = self.names_to_index[name]        
        view_index = view_id
        return self.__getitem__(mat_index * each_size + view_index)

    def summary(self, examples_per_group: int = 3) -> str:
        lines = [""]
        try:
            total_samples = len(self)
        except Exception:
            total_samples = "unknown"
        lines.append(f"\tMultipleMaterials: {self.n_mats} materials, {total_samples} total frames")

        # Use precomputed groups (cfg-driven) if available
        for group_name, idx_list in self.groups.items():
            group_items = [self.datasets[i] for i in idx_list]
            frames = group_items[0].size
            resolutions = group_items[0].resolution
            input_types = group_items[0].input_type
            tonemap_types = group_items[0].tonemapping_type
            lazy_modes = group_items[0].lazy_load
            lines.extend(
                [f"\t\t{group_name}: ",
                f"\t\t\tmaterials={len(group_items)}, ",
                f"\t\t\tn_frames (or slices) = {frames}",
                f"\t\t\tres={resolutions}, type={input_types}, tm={tonemap_types}", f"\t\t\tlazy={lazy_modes}"]
            )
            # sample a few material names from each group
            names = [item.material_name for item in group_items]
            lines.append(f"\t\t\t{', '.join(names)}")
            # remaining = len(group_items) - examples_per_group
            # if remaining > 0:
            #     lines.append(f"\t\t\t ...and {remaining} more")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.__str__()



def exclude_list(initiallist, excluded):
    if excluded is None:
        return initiallist
    return [item for item in initiallist if item not in excluded]


def get_items_list(end, start = 0, every_other = 1, specific_list = None, exclude = None):
    if specific_list:
        return exclude_list(specific_list, exclude)
    else:
        return exclude_list(list[int](range(start, end, every_other)), exclude)

def get_material_list(csv_path, end = None, start = 0, every_other = 1, specific_list = None, exclude=None):
    if specific_list and isinstance(specific_list[0], str):
        return exclude_list(specific_list, exclude)
    else:
        material_names = read_csv(csv_path, only_material_name = True)
        if end is None:
            end = len(material_names)
        material_ids = get_items_list(end, start, every_other, specific_list, exclude)
        return exclude_list( [material_names[i] for i in material_ids], exclude)


import csv

def read_csv(csv_path, only_material_name = True):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline='') as f:
        reader = csv.reader(f)

        # skip header
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue

            material_name = row[0].split('/')[0]
            prompt = row[1]

            rows.append((material_name, prompt))

    if only_material_name:
        return [row[0] for row in rows]
    else:
        return rows




def load_dataset(dataset_path, input_type, tonemapping_type, resolution, materials_info, frames_info, dataset_n_frames, lazy_load, scene_state_path, rep_frame_info, disable_rep, rep_use_latent, group_name, rep_images_path, split, traj, rep_resolution, use_hdf5=False, hdf5_data_root=None, hdf5_n_slices=64, random_frame=False):
    dataset_path = Path(dataset_path)
    frame_list = get_items_list(**frames_info) if not use_hdf5 else None
    representative_frame_list = get_items_list(**rep_frame_info) if rep_frame_info else None
    material_list = get_material_list(**materials_info)

    states = None
    if not use_hdf5:
        states = load_scene_states(Path(scene_state_path), dataset_n_frames)
        states = [states[i] for i in frame_list]

    materials_dict = {}
    
    for i, name in tqdm(enumerate[Any](material_list), total=len(material_list), desc=f'{group_name}: Loading materials'):
        if name in materials_dict:
            raise ValueError(f"Duplicate named material: {name}")

        
        materials_dict[name] = SingleMaterial(
            material_name = name, 
            material_path = dataset_path / name, 
            dataset_n_frames = dataset_n_frames, 
            resolution = resolution, 
            tonemapping_type = tonemapping_type, 
            input_type = input_type, 
            frame_list =  frame_list,
            lazy_load = lazy_load,
            representative_frame_list = representative_frame_list if not disable_rep else None,
            rep_images_use_latent = rep_use_latent,
            states = states,
            rep_images_path = rep_images_path,
            split = split,
            traj = traj,
            rep_resolution = rep_resolution,
            use_hdf5 = use_hdf5,
            hdf5_data_root = hdf5_data_root,
            hdf5_n_slices = hdf5_n_slices
            )

    return materials_dict
        


def check_duplicates(materials, added):
    common_keys = set(materials) & set(added)
    assert len(common_keys)==0, f"Duplicate materials: {common_keys}"


# VALID_SPLITS = {'train', 'test', 'val'}
VALID_TRAJS = {'dense', 'sparse', 'spiral', 'grazing'}
CONFIG_DIR = Path(__file__).parent / "config" / "datasets"


def build_configs(dataset_strings: list[str]) -> dict:
    """
    Parse strings like 'matsynth8_train_dense_all' and build config dicts.
    Format: <dataset_name>_<split>_<traj>_<material_preset>
    """
    cfgs = {}
    for ds_str in dataset_strings:
        parts = ds_str.split('_')
        assert len(parts) == 4, f"Expected format: <dataset>_<split>_<traj>_<preset>, got: {ds_str}"
        dataset_name, split, traj, preset = parts

        # assert split in VALID_SPLITS, f"Invalid split '{split}'. Must be one of {VALID_SPLITS}"
        assert traj in VALID_TRAJS, f"Invalid traj '{traj}'. Must be one of {VALID_TRAJS}"

        # Load base dataset yaml with {split} and {traj} interpolated
        with open(CONFIG_DIR / f"{dataset_name}.yaml", 'r') as f:
            raw = f.read().replace('{split}', split).replace('{traj}', traj)
        cfg = OmegaConf.create(raw)

        cfg.pop('defaults', None)
        cfg.pop('name', None)  # not used by load_dataset
        cfg['split'] = split
        cfg['traj'] = traj

        # Load material preset if not "all"
        if preset != 'all':
            preset_cfg = OmegaConf.load(CONFIG_DIR / 'presets' / 'materials' / f"{preset}.yaml")
            if 'specific_list' in preset_cfg[dataset_name]:
                cfg.materials_info.specific_list = preset_cfg[dataset_name].specific_list
            if 'exclude' in preset_cfg[dataset_name]:
                old_exclude = cfg.materials_info.exclude
                old_exclude = [] if old_exclude is None else list(old_exclude)
                cfg.materials_info.exclude = old_exclude + list(preset_cfg[dataset_name].exclude) 

        if 'lazy_load' not in cfg:
            cfg.lazy_load = True
        cfgs[ds_str] = cfg


    return cfgs


def load_datasets(tonemapping_type, resolution, cfgs, disable_rep, rep_use_latent, unrolled, rep_resolution= None, build_cfg = True, use_hdf5=False, hdf5_data_root=None, hdf5_n_slices=64, random_frame=False):
    if build_cfg:
        cfgs = build_configs(cfgs)
    materials_dict = {}
    groups_by_name: dict[str, list[str]] = {}
    for cfg_idx, (cfg_name, cfg) in enumerate(cfgs.items()):
        added_materials = load_dataset(tonemapping_type=tonemapping_type, resolution=resolution, disable_rep = disable_rep, rep_use_latent = rep_use_latent, group_name = cfg_name, rep_resolution = rep_resolution, use_hdf5=use_hdf5, hdf5_data_root=hdf5_data_root, hdf5_n_slices=hdf5_n_slices, random_frame=random_frame, **cfg)        
        added_fullname = {f'{cfg_name}***{name}': material for name, material in added_materials.items()}
        check_duplicates(materials_dict.keys(), added_fullname.keys())
        materials_dict.update(added_fullname)
        groups_by_name.setdefault(cfg_name, []).extend(added_fullname.keys())

    
    return MultipleMaterials(materials_dict, unrolled=unrolled, groups_by_name=groups_by_name)





    

        

