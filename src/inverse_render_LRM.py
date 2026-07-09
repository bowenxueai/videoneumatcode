import hydra
from omegaconf import DictConfig
from inverse_render import main as inverse_render_main




@hydra.main(version_base=None, config_path="config", config_name="inverse_render_LRM")
def main(cfg : DictConfig) -> None:
    inverse_render_main(cfg)


if __name__ == '__main__':
    main()





