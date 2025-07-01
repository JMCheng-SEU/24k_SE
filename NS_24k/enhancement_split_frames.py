import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stft import STFT
from utils.utils import initialize_config
import numpy as np
# from test_for_PSD import *
# from model.Simple_DCCRN import *
# from model.DCCRN_origin import *
# from model.oneC_DIL_CRN import DPCRN_Model
from model.oneC_DPRNN_Streamer import DPCRN_Model_Streamer, Simple_Streamer
from model.CGRNN_FB import CGRNN_FB, CGRNN_FB_Large, CGRNN_FB_Small, CGRNN_FB_Lite_noSC_new, CGRNN_FB_Lite_noSC_new_withVAD_Mag
from model.DPCRN_RT import DPCRN_Model_RT, DPCRN_Model


def main(config, epoch):
    # root_dir = Path(config["experiments_dir"]) / config["name"]
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"

    """============== 加载数据集 =============="""
    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )

    """============== 加载模型断点（"best"，"latest"，通过数字指定） =============="""
    # model = initialize_config(config["model"])
    model = DPCRN_Model()
    # device = torch.device("cuda:0")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    stft = STFT(
        filter_length=512,
        hop_length=256
    ).to('cuda')



    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix())
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to('cuda')
    model.eval()

    """============== 增强语音 =============="""
    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    for i, (mixture, _, _, names) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]
        print(name)

        mixture = mixture.to('cuda')

        mixture_D = stft.transform(mixture)
        mixture_real = mixture_D[:, :, :, 0]
        mixture_imag = mixture_D[:, :, :, 1]

        spec_complex = torch.stack([mixture_real, mixture_imag], 1)

        with torch.no_grad():
            est_real, est_imag = model(spec_complex)

        enhanced_D = torch.stack([est_real, est_imag], 3)
        enhanced = stft.inverse(enhanced_D)

        enhanced = enhanced.detach().cpu().squeeze().numpy()
        sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    # parser.add_argument("-C", "--config", type=str, required=True,
    #                     help="Specify the configuration file for enhancement (*.json).")
    # parser.add_argument("-E", "--epoch", default="best",
    #                     help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    # args = parser.parse_args()
    config_path = "D:\\JMCheng\\RT_16k\\codes\\CGRNN_FB_512_16k\\config\\enhancement\\Non-Stationary-Datasets.json"
    config = json.load(open(config_path))
    config["name"] = os.path.splitext(os.path.basename(config_path))[0]
    main(config, "best")
    # main(config, 39)
