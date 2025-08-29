# 2P Imaging Analysis Pipeline

A modular, plug-and-play pipeline for 2-photon calcium imaging data using [Suite2p](https://github.com/MouseLand/suite2p). Designed for analyzing multiple experiments with minimal setup, across different imaging planes and acquisition sessions.

## Features

- **Suite2p integration** with multi-experiment compatibility  
- **Plane-aware support** (Z-stack / multiplane data)
- **Run hashing** to ensure reproducibility of analyses
- **Postprocessing and visualization** with customizable outputs  

## Credits

- **Johannes Kappel** – Original Suite2p analysis code
- **Enrico Kohn** & **Katja Slangewal** – Classifier design  
- **Inbal Shainer** – Cellpose model  
- **Joseph Donovan** – BiDiOffset utility  

## Installation

```bash
git clone https://github.com/lisakatobauer/twop_imaging_analysis_runner.git
cd twop_imaging_analysis_runner
pip install -e .
```

## Project Structure

| Module              | Description                                           |
|---------------------|-------------------------------------------------------|
| `Suite2pProcessor`  | Prepares and runs Suite2p on multi-session data       |
| `Suite2pLoader`     | Loads processed Suite2p output                        |
| `Suite2pTraces`     | Extracts, filters, and normalizes fluorescence traces |
| `Suite2pVisualiser` | Visualizes ROI activity, top traces, and heatmaps     |

## Getting Started

Each experiment requires a Python config file (e.g., `fish_1.py`) with information on the experiments.

Example file can be found in `twop_imaging_analysis_runner/config/configlist`. 

Example runs can be found in `/usage_examples`.

The `twop_imaging_analysis_runner/config/base_config.py` file contains the pathing that should be defined by each user.

## Usage Examples

Basic pipeline execution:
```python
from twop_imaging_analysis_runner import Suite2pProcessor, Suite2pTraces

# Process Suite2p data
processor = Suite2pProcessor(config, fishnum=1)
processor.run()

# Extract and process traces
traces = Suite2pTraces(config, fishnum=1)
traces.process_all()
```

## Dependencies

- Python 3.8+
- Suite2p
- NumPy
- SciPy
- scikit-learn
- Matplotlib (for visualization)
- Cellpose (optional, for cell segmentation)

## Configuration

Create a configuration file following the examples in the config directory. Key parameters include:
- Data paths
- Experiment metadata
- Processing parameters
- Output directories

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Created by [Lisa Bauer](https://github.com/lisakatobauer). Feel free to reach out!

