import argparse
import json
import os

# parser = argparse.ArgumentParser(description='EHNET')
# parser.add_argument("-C", "--config", required=True, type=str, default="E:\JMCheng\A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement-master\config\\train\\SA-SE-CRN-Baseline.json",
#                     help="Specify the configuration file for training (*.json).")
# parser.add_argument('-D', '--device', default=True, type=str,
#                     help="Specify the GPU visible in the experiment, e.g. '1,2,3'.")
# parser.add_argument("-R", "--resume", action="store_true", default=False,
#                     help="Whether to resume training from a recent breakpoint.")
# args = parser.parse_args()

# if args.device:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from utils.utils import initialize_config
from torch.nn.utils.rnn import pad_sequence
from time_dataset import make_loader
from torch.optim.lr_scheduler import StepLR


from model.DPCRN import DPCRN_Model_512, DPCRN_Model_512_Slim, DPCRN_Model_512_Slim_extra
from model.CGRNN_FB_SC import CGRNN_FB, CGRNN_FB_Mag
from model.DB_AIAT_model.aia_trans_onemic import dual_aia_trans_merge_crm_onemic, new_aia_complex_trans_ri, new_aia_mag_trans_ri, DF_Serial_aia_complex_trans_ri
from model.AIA_DPCRN.AIA_CRN import DF_AIA_CRN_oneL

def pad_to_longest(batch):
    mixture_list = []
    clean_list = []
    names = []
    n_frames_list = []

    for mixture, clean, n_frames, name in batch:
        mixture_list.append(torch.tensor(mixture).reshape(-1, 1))
        clean_list.append(torch.tensor(clean).reshape(-1, 1))
        n_frames_list.append(n_frames)
        names.append(name)

    # seq_list = [(L_1, 1), (L_2, 1), ...]
    #   item.size() must be (L, *)
    #   return (longest_len, len(seq_list), *)
    mixture_list = pad_sequence(mixture_list).squeeze(2).permute(1, 0)
    clean_list = pad_sequence(clean_list).squeeze(2).permute(1, 0)

    return mixture_list, clean_list, n_frames_list, names


def collate_fn(data):
    inputs, s1 = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    s1 = np.array(s1, dtype=np.float32)
    # inputs_vad = np.array(inputs_vad, dtype=np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(s1)

def main(config, resume):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.enabled = False

    print('preparing data...')
    scp_file_path = "D:\\JMCheng\\RT_24k\\24k_SE_exp\\133h_24k_AntTTS_SE.lst"
    train_dataset = make_loader(
        scp_file_path
            )
    # train_dataset = initialize_config(config["train_dataset"])
    train_data_loader = DataLoader(
        shuffle=config["train_dataloader"]["shuffle"],
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True
    )

    validation_dataset = initialize_config(config["validation_dataset"])
    valid_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"],
        collate_fn=pad_to_longest,
        shuffle=config["validation_dataloader"]["shuffle"]
    )

    # model = initialize_config(config["model"])
    model = DF_Serial_aia_complex_trans_ri()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        # params=model.get_params(weight_decay=0.001),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], 0.999)
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        train_dataloader=train_data_loader,
        validation_dataloader=valid_data_loader
    )


    trainer.train()


if __name__ == '__main__':
    # load config file
    config_path = "D:\\JMCheng\\RT_24k\\codes\\NS_24k\\config\\train\\PHA-CRN.json"
    config = json.load(open(config_path))
    config["experiment_name"] = os.path.splitext(os.path.basename(config_path))[0]
    config["train_config_path"] = config_path

    main(config, resume=False)
