from twop_imaging_analysis_runner.core.suite2p_processor import classifier_file, Suite2pProcessor
from twop_imaging_analysis_runner.core import Suite2pTraces
from twop_imaging_analysis_runner.config.config import Suite2pConfig
from twop_imaging_analysis_runner.utils.utils import get_git_root

from twop_imaging_analysis_runner.config.base_config import raw_path, processed_path

# 1. Config
config_path = get_git_root() / 'twop_imaging_analysis_runner' / 'config' / 'configlist'

suite2p_ops = {'framerate': 30.0, 'nplanes': 6, 'classifier_path': classifier_file}

config = Suite2pConfig(
    config_path,
    raw_path,
    processed_path,
    suite2p_ops)

config.get_fish_config(120)
# config.load_all_fish_configs()
fish_ids = config.get_loaded_fish_ids()

# 2. Initialize and run the processor

for fishnum in fish_ids:
    processor = Suite2pProcessor(config, fishnum)
    processor.run_extraction()

# 3. Do some standard processing on the traces

for fishnum in fish_ids:
    traces = Suite2pTraces(config, fishnum)
    traces.process_all()
