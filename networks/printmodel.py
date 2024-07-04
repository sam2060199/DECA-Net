import argparse
import logging
import os
import random
import numpy as np
import torch

import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

from networks.vision_transformer import SwinUnet
from torchsummary import summary
from config import get_config

config = get_config()

model = SwinUnet(config, 256, 2).cuda()
batch_size=128
summary(model,input_size=(batch_size,3,256,256))