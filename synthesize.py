import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser
import numpy as np
import torchaudio

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "model_best.pth"


def main(config, input_folder, out_folder):
    input_folder = Path(input_folder)
    out_folder = Path(out_folder)

    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)

    for file in tqdm(enumerate(input_folder.iterdir())):
        melspec = torch.from_numpy(np.load(file)).to(device)
        
        model.eval()
        output = model()
        audio = output["prediction"][0].cpu().detach()
        new_file_name = file.stem + ".wav"
        out_file_name = out_folder/ new_file_name
        torchaudio.save(out_file_name, audio)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="results",
        type=str,
        help="Dir to write results",
    )
    args.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="Path to dir with mels to synthesize",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    main(config, args.input, args.output)