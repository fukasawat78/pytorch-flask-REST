import os
import torch
from torchvision import models
import torchvision.transforms as transforms

dir_path = './static/models/densnet_state_dict/'

if not os.path.exists(dir_path):
    os.mkdir(dir_path) 

model = models.densenet121(pretrained=True)

file_path = dir_path + "model"
torch.save(model.state_dict(), file_path)