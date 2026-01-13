from twop_imaging_analysis_runner.core import Suite2pLoader
from twop_imaging_analysis_runner.core.suite2p_processor import classifier_file
from twop_imaging_analysis_runner.core.suite2p_visualiser import Suite2pVisualiser
from twop_imaging_analysis_runner.config.config import Suite2pConfig
from twop_imaging_analysis_runner.utils.utils import get_git_root

from twop_imaging_analysis_runner.config.base_config import raw_path, processed_path

# 1. Config
config_path = get_git_root() / 'config' / 'configlist'

suite2p_ops = {'framerate': 30.0, 'nplanes': 6, 'classifier_path': classifier_file}

config = Suite2pConfig(
    config_path,
    raw_path,
    processed_path,
    suite2p_ops)

# config.load_all_fish_configs()
config.get_fish_config(125)
fish_ids = config.get_loaded_fish_ids()
experiment_n = 1

# 2. Initialize and run the loader

for fishnum in fish_ids:
    loader = Suite2pLoader(config, fishnum, experiment_n, suite2p_ops['nplanes'])
    data = loader.get_basic_data()

# 3. Do some data visualization

    visualiser = Suite2pVisualiser(data, config, fishnum, experiment_n)
    visualiser.plot_highly_active(save_plot=True)
    visualiser.plot_location(save_plot=True)
    visualiser.plot_heatmap(save_plot=True)

