'''

Leaf Vein Segmentation based on U-Net or Resnetbased-FCN
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# dataload.function

'''
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image as Image

def data_list(path):
    file_list = os.listdir(path)
    data_list = []
    for i in range(0, len(file_list)):
        data_path = os.path.join(path, file_list[i])
        if os.path.isfile(data_path):
            data_list.append(data_path)
    return data_list

class MaskDataset(Dataset):
    def __init__(self, path, phase = 'train', input_size=572, output_size=388):

        self.input_images = data_list(os.path.join(path, 'input')) #得到path+input下所有图片列表
        self.target_masks = data_list(os.path.join(path, 'mask'))

        self.phase = phase

        self.input_label_data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                #transforms.RandomResizedCrop(input_size),
                #transforms.RandomHorizontalFlip(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                #transforms.CenterCrop(input_size),
            ]),
        }
        self.input_data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.mask_data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(output_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(output_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
        }
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        
        image_path = self.input_images[idx]
        image = Image.open(image_path)
        mask_path = self.target_masks[idx]
        mask = Image.open(mask_path)

        image = self.input_data_transforms[self.phase](self.input_label_data_transforms[self.phase](image))
        mask = self.mask_data_transforms[self.phase](self.input_label_data_transforms[self.phase](mask))

        return [image, mask]

