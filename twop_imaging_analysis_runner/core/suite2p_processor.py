import glob
import os
import shutil
import sys
from typing import Dict

import numpy as np

import skimage
from skimage import io
from skimage.filters import gaussian
import tifffile as tiff
from suite2p import run_s2p, default_ops, classification
from suite2p.io import BinaryFile
from suite2p.registration.register import shift_frames_and_write

from twop_imaging_analysis_runner.utils import utils
from twop_imaging_analysis_runner.config.config import Suite2pConfig
from twop_imaging_analysis_runner.utils.processingunit import ProcessingUnit
from twop_imaging_analysis_runner.usage_examples.data.standard_ops import standard_ops
from twop_imaging_analysis_runner.utils.utils import output_dir

classifier_file = os.path.join(output_dir, 'EK_classifier16012020.npy')


class Suite2pProcessor(ProcessingUnit):
    """
    Runs suite2p pipeline across experiments in stages from python. Perform ROI detection on downsampled frames.
    """

    def __init__(self, config: Suite2pConfig, fishnum):

        """
        Initialize Suite2p processing pipeline.

        Args:
            config: instance of config
            fishnum: your fishnum
        """

        super().__init__(config.processed_path)
        self.config = config
        self.your_ops = standard_ops
        self.suite2ppath_raw = config.raw_path
        self.suite2ppath_processed = config.processed_path

        self.setup_data = config.suite2p_ops
        self.framerate = self.setup_data.get('framerate', 30)
        self.nplanes = self.setup_data.get('nplanes', 1)
        self.downsampling_factor = self.setup_data.get('downsampling_factor', 5)
        self.classifierfile = self.setup_data.get('classifierfile', None)
        self.bidirectional_scanning = self.setup_data.get('bidirectional_scanning', True)

        self.fishnum = fishnum
        self.fishdata = config.get_fish_config(fishnum)
        self.experiments = self.fishdata['experiments']
        self.experiment_lengths = self.fishdata['experiment_lengths']

        # Derived parameters
        self.volumerate = self.framerate
        self.downsample_volumerate = self.volumerate / self.downsampling_factor
        self.date = list(self.experiments.keys())[0]

        # Initialize paths
        self.downsampled_filename = None
        self.downsampled_filepath = None
        self.filter_path = None
        self.plane_save_path = None
        self.plane = None
        self.save_path = os.path.join(self.suite2ppath_processed, f'Fish_{self.fishnum}', 'suite2p')

    def run_extraction(self):
        """ Run the complete Suite2p processing pipeline. """
        os.makedirs(self.save_path, exist_ok=True)

        if self.nplanes > 1:
            # In the case of multi-plane imaging, splitting those planes is required.
            self.splitting_files_by_plane()

        for plane_n in range(self.nplanes):
            self.plane = plane_n
            self.plane_save_path = os.path.join(self.save_path, f'plane{self.plane}')

            # Skip if already processed
            if os.path.exists(os.path.join(self.plane_save_path, 'suite2p', 'plane0', 'stat.npy')):
                print(f'{self.date} fish {self.fishnum} plane {self.plane} was already processed. Skipping.')
                continue

            self.make_binary()
            self.filtering_images()
            self.motioncorrection()
            self.apply_motioncorrection_toraw()
            self.downsample()
            self.roi_detection()

            if self.classifierfile:
                self.apply_classifier()

        self.split_experiments_and_save_outputs()

    def run_extraction_hashed(self):
        # Check for existing matching run
        existing_run = self.find_matching_run(self.config)
        if existing_run:
            print(f"Using existing run with hash {existing_run}")
            self.run_hash = existing_run
            return

        # Setup new run
        self.run_hash = self.setup_run(self.config)
        self.current_ops = self._get_full_ops()

        # Rest of your processing pipeline...
        self.save_path = self.get_run_folder()
        self.run_extraction()

    def splitting_files_by_plane(self):
        """
        Split multi-plane TIF files from experiments into per-plane files.

        Creates directories for each plane and saves all recordings for a given plane
        across all experiments.
        """
        # Calculate expected file counts
        total_file_count = sum(
            len(os.listdir(os.path.join(self.suite2ppath_raw, self.date, f'Fish_{self.fishnum}', f'{int(exp_n)}')))
            for exp_n in self.experiments[self.date].keys()
        )
        total_expected_file_count = total_file_count * self.nplanes

        # Check existing files
        existing_files = self._get_existing_plane_files(self.save_path)

        if len(existing_files) >= total_expected_file_count:
            print('All plane files already processed. Skipping plane splitting.')
            return

        print('Starting plane splitting for experiment files...')
        counter = 0

        for experiment_n in self.experiments[self.date].keys():
            experiment_raw_path = os.path.join(
                self.suite2ppath_raw, self.date, f'Fish_{self.fishnum}', f'{int(experiment_n)}'
            )
            experiment_files = [file for file in os.listdir(experiment_raw_path) if file.endswith('.tif')]
            plane_arr = np.arange(self.nplanes)

            for file_n, file in enumerate(experiment_files):
                file_path = os.path.join(experiment_raw_path, file)
                print(f'Processing file {file_n + 1}/{len(experiment_files)}: {file}')

                try:
                    with tiff.TiffFile(file_path) as tif_instance:
                        tif_data = np.array([page.asarray() for page in tif_instance.pages])

                    roll = tif_data.shape[0] % self.nplanes
                    self._process_planes(tif_data, plane_arr, experiment_n, counter, self.save_path,
                                         existing_files, file)

                except Exception as e:
                    print(f'Error processing file {file_path}: {e}')
                    continue

                counter += 1
                plane_arr = np.roll(plane_arr, -roll)

        print('Completed splitting files by planes.')

    @staticmethod
    def _get_existing_plane_files(path: str) -> set:
        """Get set of existing (plane, recording) tuples."""
        all_plane_files = glob.glob(os.path.join(path, '**', 'merge_exp*_plane*_rec*_raw.tif'), recursive=True)
        return {
            (int(os.path.basename(f).split('plane')[1].split('_')[0]),
             int(os.path.basename(f).split('rec')[1].split('_')[0]))
            for f in all_plane_files
        }

    def _process_planes(
            self,
            tif_data: np.ndarray,
            plane_arr,
            experiment_n: str,
            counter: int,
            base_path: str,
            existing_files: set,
            file
    ) -> None:
        """Process and save individual planes from a multi-plane TIFF."""
        plane_tifs = [[] for _ in range(self.nplanes)]
        for plane_idx in range(self.nplanes):
            if (plane_idx, counter) in existing_files:
                print(f'Skipping plane {plane_idx} for file {file} (already processed)')
                continue

            frames = tif_data[plane_idx::self.nplanes]

            if self.bidirectional_scanning:
                frames = utils.bidi_offset_correction_plane(frames)

            plane_tifs[plane_arr[plane_idx]].append(frames.astype('int16'))

        for plane_idx, ts_plane in enumerate(plane_tifs):
            if not ts_plane:
                continue

            ts_full = np.concatenate(ts_plane, axis=0)

            plane_save_path = os.path.join(base_path, f'plane{plane_idx}')
            os.makedirs(plane_save_path, exist_ok=True)

            save_file_path = os.path.join(
                plane_save_path,
                f'merge_exp{experiment_n}_plane{plane_idx}_rec{counter}_raw.tif'
            )

            try:
                io.imsave(save_file_path, ts_full.astype('int16'), check_contrast=False, plugin='tifffile')
                print(f"Saved plane {plane_idx} recording {counter} to {save_file_path}")
            except OSError:
                try:  # sometimes saving fails when saving to network, because of connection issues. retry.
                    io.imsave(save_file_path, ts_full.astype('int16'), check_contrast=False, plugin='tifffile')
                    print(f"Saved plane {plane_idx} recording {counter} to {save_file_path}")
                except Exception as e:
                    print(f"Error saving file {save_file_path}: {e}")
            except Exception as e:
                print(f"Error saving file {save_file_path}: {e}")

    def make_binary(self) -> None:
        """Convert TIFF files to binary format for Suite2p."""
        if os.path.exists(os.path.join(self.plane_save_path, 'suite2p', 'plane0', 'data.bin')):
            print('Binary data bin already exists, skipping...')
            return

        ops = default_ops()
        if self.your_ops:
            ops = standard_ops()

        db = {
            'look_one_level_down': False,
            'data_path': [self.plane_save_path],
            'roidetect': False,
            'do_registration': False,
        }
        print('Starting make binary...')
        run_s2p(ops=ops, db=db)

    def filtering_images(self, sigma: float = 4) -> None:
        """Apply Gaussian filter to images."""
        self.filter_path = os.path.join(self.plane_save_path, 'filtered')
        os.makedirs(self.filter_path, exist_ok=True)

        plane_files = glob.glob(os.path.join(self.plane_save_path, f'merge_exp*_plane{self.plane}_rec*_raw.tif'))
        filtered_files = [
            os.path.join(self.filter_path, os.path.basename(f).replace('raw.tif', 'raw_filt.tif'))
            for f in plane_files
        ]

        # Are there already filtered files in the filter_path folder?
        if len(plane_files) == len(glob.glob(os.path.join(self.filter_path, f'*plane{self.plane}*raw_filt.tif'))):
            print('Plane files already filtered, skipping...')
            return

        print('Starting filtering images...')
        for plane_file, filtered_file in zip(plane_files, filtered_files):
            if os.path.exists(filtered_file):
                continue

            try:
                ts = tiff.imread(plane_file)
                tsg = gaussian(ts, sigma=(0, sigma, sigma), mode='nearest', preserve_range=True)
                io.imsave(filtered_file, tsg.astype('int16'), check_contrast=False, plugin='tifffile')
                print(f"Saved filtered file: {filtered_file}")
            except Exception as e:
                print(f"Error processing {plane_file}: {e}")
                continue

        # Verify that all filtering was succesful
        if len(plane_files) == len(glob.glob(os.path.join(self.filter_path, f'*plane{self.plane}*raw_filt.tif'))):
            print(f"All filtered files for plane {self.plane} are processed and available.")
        else:
            print(f"Failed to create all filtered files for plane {self.plane}. Exiting.")
            sys.exit()

    def motioncorrection(self) -> None:
        """Perform motion correction on filtered images."""
        if os.path.exists(os.path.join(self.filter_path, 'suite2p', 'plane0', 'ops.npy')):
            print('Motion correction already finished, skipping...')
            return

        print('Starting motion correction...')

        ops = default_ops()
        if self.your_ops:
            ops = standard_ops()
        db = {
            'look_one_level_down': False,
            'data_path': [self.filter_path],
            'reg_tif': True,
            'roidetect': False,
            'do_registration': True,
            'nonrigid': True,
            'keep_movie_raw': True,
            'do_regmetrics': True
        }
        run_s2p(ops=ops, db=db)

    def apply_motioncorrection_toraw(self) -> None:
        """Apply motion correction to raw data."""
        if os.path.exists(os.path.join(self.filter_path, 'suite2p', 'plane0', 'reg_tif_chan2')):
            print('Motion correction already applied to raw tif files, skipping...')
            return

        # Load the ops.npy file that contains the motion correction offsets
        ops_path = os.path.join(self.filter_path, 'suite2p', 'plane0', 'ops.npy')
        filt_ops = np.load(ops_path, allow_pickle=True).item()

        # Define paths to the raw and registered binary files
        binfile_raw = os.path.join(self.plane_save_path, 'suite2p', 'plane0', 'data.bin')
        binfile_reg = os.path.join(self.filter_path, 'suite2p', 'plane0', 'reg.bin')

        print('Starting to apply motion correction to raw tif files...')
        # Open the raw binary file for reading
        with BinaryFile(Ly=filt_ops['Ly'], Lx=filt_ops['Lx'], filename=binfile_raw) as f_raw:
            # Open the registered binary file for writing
            with BinaryFile(Ly=filt_ops['Ly'], Lx=filt_ops['Lx'],
                            filename=binfile_reg, n_frames=f_raw.n_frames) as f_reg:
                # Update options to ensure the registration TIFFs are saved
                filt_ops["reg_tif_chan2"] = True
                filt_ops["save_path"] = os.path.join(self.filter_path, 'suite2p', 'plane0')

                # Apply motion correction shifts and write the registered frames
                shift_frames_and_write(
                    f_alt_in=f_raw,
                    f_alt_out=f_reg,
                    yoff=filt_ops['yoff'],
                    xoff=filt_ops['xoff'],
                    yoff1=filt_ops.get('yoff1', None),
                    xoff1=filt_ops.get('xoff1', None),
                    ops=filt_ops
                )

    # downsampling the data for ROI extraction
    def downsample(self) -> None:
        """Downsample motion-corrected data."""
        self.downsampled_filepath = os.path.join(self.plane_save_path, f'downsampled')
        self.downsampled_filename = os.path.join(
            self.downsampled_filepath,
            f'downsampled_{self.downsample_volumerate:.2f}Hz_registered_plane{self.plane}.tif')

        # Check if the downsampled file already exists to avoid redundant processing
        if os.path.exists(self.downsampled_filename):
            print('Downsampling already performed, skipping...')
            return

        # Source directory for registered tif files
        datapath = os.path.join(self.filter_path, 'suite2p', 'plane0', 'reg_tif_chan2')
        mcfiles = sorted(glob.glob(os.path.join(datapath, 'file*')))

        print("Starting downsampling...")
        ts_mc = []
        # Process each motion-corrected file
        for idx, mcfile in enumerate(mcfiles):
            try:
                with tiff.TiffFile(mcfile, is_scanimage=False) as ts_file:
                    # Load the full time series for each frame
                    ts = np.array([i.asarray() for i in ts_file.pages])
                # Downsample by averaging across specified factor
                ts_ds = np.array([
                    np.mean(ts[i:i + self.downsampling_factor], axis=0)
                    for i in range(0, ts.shape[0], self.downsampling_factor)
                ])
                del ts
                # Concatenate downsampled data for each file
                ts_mc.append(ts_ds)
            except Exception as e:
                print(f"Error processing file {mcfile}: {e}")
                continue

        # Concatenate all downsampled frames and save to specified path
        if ts_mc:
            os.makedirs(os.path.dirname(self.downsampled_filename), exist_ok=True)
            ts_mc = np.concatenate(ts_mc, axis=0).astype('int16')
            print(f"Saving downsampled file to {self.downsampled_filepath}")
            skimage.io.imsave(self.downsampled_filename, ts_mc, check_contrast=False, plugin='tifffile')
        del ts_mc

    def roi_detection(self) -> None:
        """Detect ROIs in the downsampled data."""
        if os.path.exists(os.path.join(self.plane_save_path, 'suite2p', 'plane0', 'stat.npy')):
            print("ROI detection already done, skipping...")
            return

        print('Starting ROI detection...')
        ops = default_ops()
        if self.your_ops:
            ops = standard_ops()

        db = {
            'look_one_level_down': False,
            'data_path': [os.path.join(self.filter_path, 'suite2p', 'plane0', 'reg_tif_chan2')], # TODO downsampled
            'save_path0': self.plane_save_path,
            'fs': self.volumerate,
            'reg_tif': False,
            'roidetect': True,
            'do_registration': False
        }
        run_s2p(ops=ops, db=db)

    def apply_classifier(self) -> None:
        """Apply cell classifier to detected ROIs."""
        if os.path.exists(os.path.join(self.plane_save_path, 'suite2p', 'plane0', 'iscell_classifier.npy')):
            print("Classifier already applied, skipping...")
            return

        print("Starting applying classifier...")
        stat = np.load(os.path.join(self.plane_save_path, 'suite2p/plane0/stat.npy'), allow_pickle=True)
        iscell = classification.Classifier(self.classifierfile, keys=['npix_norm', 'compact', 'skew']).run(stat)
        np.save(os.path.join(os.path.join(self.plane_save_path, 'suite2p/plane0'), 'iscell_classifier.npy'), iscell)

    def split_experiments_and_save_outputs(self) -> None:
        """Split and save fluorescence data by experiment."""
        experiment_ns = list(self.experiments[self.date].keys())
        last_experiment = os.path.join(
            self.suite2ppath_processed, f'Fish_{self.fishnum}', f'{int(experiment_ns[-1])}', 'suite2p',
            f'plane{self.nplanes - 1}', 'spks.npy'
        )

        if os.path.exists(last_experiment):
            print("Splitting files by experiment already done, skipping...")
            return

        print("Starting splitting files by experiment...")

        for plane in range(self.nplanes):
            plane_f_path = os.path.join(self.save_path, f'plane{plane}', 'suite2p', 'plane0', 'F.npy')
            if not os.path.exists(plane_f_path):
                print(f"Warning: File {plane_f_path} does not exist. Skipping plane {plane}.")
                continue

            # Load fluorescence data for the plane
            F = np.load(plane_f_path)
            time_offset = 0

            n_rois, n_timepoints = F.shape
            print(f"Loaded F.npy for plane {plane}: {n_rois} ROIs, {n_timepoints} timepoints.")

            for exp_n, experiment_type in enumerate(self.experiment_lengths):
                exp_length = self.experiment_lengths[experiment_type]  # Experiment length in seconds
                experiment_n = experiment_ns[exp_n]

                # Calculate frames for the current experiment
                exp_frames = int(exp_length * self.framerate)

                F_exp = F[:, time_offset:time_offset + exp_frames]
                time_offset += exp_frames

                print(
                    f"Processing experiment {experiment_type} for plane {plane}: {exp_frames} frames from "
                    f"{time_offset} to {time_offset + exp_frames}.")

                exp_save_path = os.path.join(
                    self.suite2ppath_processed,
                    f'Fish_{self.fishnum}', f'{int(experiment_n)}', 'suite2p', f'plane{plane}')
                os.makedirs(exp_save_path, exist_ok=True)

                np.save(os.path.join(exp_save_path, f'F.npy'), F_exp)

                print(
                    f"Saved fluorescence data for experiment {experiment_n}, plane {plane}.")

                # Copy other necessary files
                for f in ['stat.npy', 'iscell.npy', 'ops.npy', 'spks.npy']:
                    src = os.path.join(self.save_path, f'plane{plane}', 'suite2p', 'plane0', f)
                    dst = os.path.join(exp_save_path, f)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

    def _get_ops_fingerprint(self) -> Dict:
        """Get a simplified version of ops settings for hashing."""
        ops = default_ops()
        if hasattr(self, 'your_ops'):
            ops.update(self.your_ops)
        return {
            'fs': self.volumerate,
            'nplanes': self.nplanes,
            'nchannels': ops.get('nchannels'),
            'tau': ops.get('tau'),
            'nonrigid': ops.get('nonrigid', True),
            'bidirectional': self.bidirectional_scanning
        }

    def _get_full_ops(self) -> Dict:
        """Get the complete ops configuration."""
        ops = default_ops()
        if hasattr(self, 'your_ops'):
            ops.update(self.your_ops)
        return ops
