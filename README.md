# NV HW4
#### Implemented by: Pistsov Georgiy 202

You can find report here: [wandb report](https://wandb.ai/goshanice/nv_project/reports/-DLA-NV-Homework--Vmlldzo2MTUxNDI3)

## Installation guide

Current repository is for Linux

(optional, not recommended) if you are trying to install it on macos run following before install:
```shell
make switch_to_macos
```

Then you run:

```shell
make install
```

## Download checkpoint:

```shell
make download_checkpoint
```
The file "model_best.pth" will be in default_test_model/

## Train model:

```shell
make train
```
Config for training you can find in src/config.json


## Synthesize something:

The melspectrograms to synthesize should be in "test_data_folder/"

```shell
make synthesize
```

The results will be in "results/"


## Run any other python script:

If you want to run any other custom python script, you can just start it with "poetry run"
For example:

Instead of:

```shell
python train.py -r default_test_model/model_best.pth
```

You can use:

```shell
poetry run python train.py -r default_test_model/model_best.pth
```

## How to train my model

```shell
poetry run python train.py -c src/config.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.