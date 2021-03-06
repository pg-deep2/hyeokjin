import os

import random
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from trainer import Trainer

from data_loader import get_loader

def main(config):
    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataroot = config.dataroot
    h_dataroot = os.path.join(dataroot,"HV")
    r_dataroot = os.path.join(dataroot,"RV")


    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, r_loader = get_loader(h_dataroot, r_dataroot, 1)

    trainer = Trainer(config, h_loader, r_loader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)