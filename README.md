# AEM-Net

Code, data, pretrained models, and figure assets accompanying the AEM-Net research project.

## Repository structure

- `data/`: example HDF5 and SAC waveform data
- `data_augument/`: scripts for generating detection and localization data
- `model/`: pretrained detection and localization models
- `monitor/`: attention modules and continuous monitoring code
- `plot/`: figures, plotting scripts, and analysis outputs

## Pretrained models

The pretrained model files are stored with Git Large File Storage (Git LFS):

- `model/detect_model.hdf5`
- `model/locate_model.hdf5`

Clone this repository with Git LFS installed to download the model contents.

## Python dependencies

The code uses Python packages including:

- TensorFlow / Keras
- NumPy
- SciPy
- h5py
- ObsPy
- Matplotlib
- pandas
- scikit-learn
- tqdm

Exact package versions depend on the environment used for the accompanying study.

## Usage

The main continuous-monitoring implementation is located at:

```text
monitor/continuous_monitor_nor_final_avg.py
```

Data-generation scripts are located in `data_augument/`, and figure-generation scripts and outputs are under `plot/`.

## Citation

Citation information will be added when the accompanying paper is publicly available.

## License

This project is released under the MIT License. See `LICENSE` for details.
