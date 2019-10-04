
from pathlib import Path
BASE_DIR = Path('.')
config = {
    'data_dir': BASE_DIR / 'dataset/lcqmc',
    'log_dir': BASE_DIR / 'outputs',
    'figure_dir': BASE_DIR / "outputs",
    'checkpoint_dir': BASE_DIR / "outputs",
    'result_dir': BASE_DIR / "outputs",
    'bert_dir':BASE_DIR / 'outputs/pytorch_pretrain'
}

