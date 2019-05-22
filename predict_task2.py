'''

Leaf Vein Segmentation based on U-Net or FCN
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# Network Prediction.function (Single image input)

'''
import torch
import PIL.Image as Image
import numpy as np
from UNet_Adapted import UNet_Adapted
from U_Net import U_Net
from FCN import FCN
from torchvision import transforms, models
import matplotlib.pyplot as plt

image_path = './test.jpg'
input_size = 224
output_size = 224
n_Classes = 1
input_data_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

image = input_data_transforms(Image.open(image_path))
#image.save('./test_mask.png')

image = image.unsqueeze(0) # for 4 channel input

model_ft = models.resnet50(pretrained=None)
model = FCN(model_ft, n_Classes, input_size) # UNet_Adapted(3, n_Classes)
model.load_state_dict(torch.load('../model/FCN_model_task2.pkl'))

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = image.cuda()
# Send the model to GPU
model = model.to(device)

# Output 1*1*388*388

mask_pred = model(image)
output_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ])

mask_pred = mask_pred[:1,:,:,:].squeeze(0).cpu()
mask_pred = output_transforms(mask_pred)
# mask_pred = (mask_pred > 0.5).float()
# scipy.misc.imsave('./test_img_mask.png',mask_pred)

mask_pred.save('./test_img_mask.png')