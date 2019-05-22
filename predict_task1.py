'''

Leaf Classification with finetuning or feature extraction on pretrained resnet-50
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# Network Prediction.function

'''

import torch
import torch.nn as nn
import PIL.Image as Image
import numpy as np
import math
from U_Net import U_Net
from torchvision import transforms, models

image_path = './test.jpg'
input_size = 224
num_classes = 3

input_data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image = input_data_transforms(Image.open(image_path))
image = image.unsqueeze(0) # for 4 channel input

model = models.resnet50()
num_ftrs = model.fc.in_features # (fc): Linear(in_features=2048, out_features=1000, bias=True)
model.fc = nn.Linear(num_ftrs, num_classes) # replace fc with 2048 to num_class

model.load_state_dict(torch.load('../model/model_task1.pkl'))

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = image.cuda()
# Send the model to GPU
model = model.to(device)
confidence_all = model(image)[0].cpu().detach().numpy()
confidence = math.exp(np.max(confidence_all)) / (math.exp(confidence_all[0])+math.exp(confidence_all[1])+math.exp(confidence_all[2]))
print(' Class is {}, with confidence {}' .format(np.argmax(confidence_all), confidence))