# emmamba

Official implementation of **emmamba** for video-based echocardiography segmentation.

This repository provides the training and evaluation code used for 3D video segmentation experiments on EchoNet-Dynamic, CAMUS, and pediatric echocardiography datasets.

## Highlights

- 3D echocardiography segmentation training framework based on PyTorch Lightning.
- Moore-curve based Mamba encoder for spatio-temporal feature modeling.
- Training scripts for EchoNet-Dynamic, CAMUS, and pediatric echo datasets.

## Requirements

The code was developed with Python 3.8 and PyTorch 1.12.1. The main environment can be created from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate EM_Mamba
```

emmamba depends on selective state-space model packages. Please install the following versions before training emmamba:

```bash
pip install causal-conv1d==1.0.0
pip install mamba-ssm==1.2.0
```

If your CUDA or PyTorch version differs from the provided environment, please install the matching wheels for `causal-conv1d` and `mamba-ssm`.

## Repository Structure

```text
.
|-- simlvseg/
|   |-- model/                 # Network architectures, including emmamba
|   `-- seg_3d/                # 3D segmentation dataset, modules, and preprocessing
|-- scripts/
|   |-- seg_3d/                # Training, testing, and inference scripts
|   |-- camus/                 # CAMUS preprocessing and evaluation utilities
|   `-- pediatric/             # Pediatric echo preprocessing utilities
|-- evaluate_bootstrap.py      # Bootstrap evaluation script
|-- environment.yml            # Conda environment
`-- LICENSE.txt
```

## Data Preparation

Prepare each dataset according to its official release instructions and update `--data_path` in the training commands.

- **EchoNet-Dynamic**: the root directory should contain the official files such as `Videos/`, `FileList.csv`, and `VolumeTracings.csv`.
- **CAMUS**: use the CAMUS dataset root or the processed CAMUS directory, depending on the experiment.
- **Pediatric echo**: use the processed A4C dataset directory.

The training scripts use RGB frame normalization with:

```text
mean = 0.12741163 0.1279413 0.12912785
std  = 0.19557191 0.19562256 0.1965878
```

## Training

### emmamba

emmamba is registered in the training framework through the `--encoder "emmamba"` option.

Before training, install:

- `causal-conv1d==1.0.0`
- `mamba-ssm==1.2.0`

### EchoNet-Dynamic

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path $$ \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "emmamba" \
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60 \
    --val_check_interval 1 \
    --seed 42
```

### CAMUS

Train on the CAMUS dataset root:

```bash
python scripts/seg_3d/seg_3d_camus_train.py \
    --data_path $$ \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "emmamba" \
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60 \
    --val_check_interval 1 \
    --seed 42
```


### Pediatric Echo

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path $$ \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "emmamba" \
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60 \
    --val_check_interval 0.25 \
    --seed 42
```


## Checkpoints

Training uses PyTorch Lightning checkpointing and monitors validation Dice score (`val_dsc`). The best checkpoint is selected automatically during testing at the end of training.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{emmamba,
  title   = {emmamba: [Please replace with the full paper title]},
  author  = {[Please replace with author list]},
  journal = {[Please replace with venue]},
  year    = {[Please replace with year]}
}
```

## License

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International License. See `LICENSE.txt` for details.

## Acknowledgements

This codebase builds on PyTorch, PyTorch Lightning, MONAI, segmentation-models-pytorch, and Mamba SSM. We thank the maintainers of these open-source projects and the providers of the EchoNet-Dynamic and CAMUS datasets.
