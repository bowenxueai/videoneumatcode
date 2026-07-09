import hydra
from omegaconf import DictConfig, OmegaConf
from render import StreamingBTFRenderer
from inverse_render import validation_cameras
import os
from pathlib import Path
import drjit as dr

@hydra.main(version_base=None, config_path="config", config_name="generate_data")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"Output directory  : @@@{output_path}@@@")


    renderer = StreamingBTFRenderer(resolution=cfg.resolution, only_center_mesh_instance=cfg.only_center_mesh_instance, arealight=cfg.area_light)
    val_states = validation_cameras(lightgeight = renderer.lightgeight, area_light_radius = renderer.area_light_radius, camera_height = renderer.camera_height, base_light_intensity = renderer.base_light_intensity, camera_radius = renderer.camera_radius, use_area_light=cfg.area_light)

    # generate dataset
    renderer.use_optix_denoiser = cfg.optix_denoiser




    states = {'train': None, 'test': val_states}
    n_frames= {'train': cfg.n_frames, 'test': len(val_states)}
    print(f"Processing {cfg.splits} splits")
    for split in cfg.splits:
        print(f"Processing {split} split")
        renderer.process_local_dataset(None,
            output_path / split,
            n_frames=n_frames[split],
            max_materials=1, 
            start_idx=0, 
            save_exr=cfg.save_exr, 
            save_frames=cfg.save_frames, 
            process_mode=1,
            material_name=cfg.material_name,
            rotations=[0], 
            arbitrary_maps=cfg.arbitrary_maps,
            val_states=states[split],
            spp = cfg.spp
            )
    dr.flush_malloc_cache()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()




    return


if __name__ == "__main__":
    main()