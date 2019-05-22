'''

Leaf Vein Segmentation based on U-Net or Resnetbased-FCN
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

# main.function

'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from dataset_task2 import MaskDataset
from UNet_Adapted import UNet_Adapted
from U_Net import U_Net
from FCN import FCN
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# hyperparameter
# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
# structure in image-mask pair:
# root/train/input
# root/train/mask
# root/val/input
# root/val/mask

data_dir = "../data/Task2"

# Models to choose from [U_Net, FCN, UNet_Adapted]
model_name = "FCN"

# Batch size for training (change depending on how much memory you have)
batch_size = 6

# Number of epochs to train for
num_epochs = 100

# Input Channel
n_Channels = 3

# Output Class
n_Classes = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# pretrained_model
use_pretrained = False
pre_model_path = None

def train_model(model, dataloaders, criterion, optimizer, lr_decay = None, num_epochs=15):
    since = time.time()

    # validation accuracy
    val_acc_history = []
    train_acc_history = []
    # for save the best accurate model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1e4

    for epoch in tqdm(range(num_epochs), ncols=70):
        print('\n [*] Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        # In fact the input has two dataloader(one for train and one for test)
        for phase in ['train', 'val']:
            print(' [**] Begin {} ...'.format(phase))
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            # Another way: dataIter = iter(dataloaders[phase]) then next(dataIter)
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # notice the validation set will run this with block but do not set gradients trainable
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    outputs_flat = outputs.view(-1)
                    targets_flat = targets.view(-1)
                    loss = criterion(outputs_flat, targets_flat)

                    # backward + optimize parameters only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(' {} Epoch Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'train' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_loss)
                train_epoch_loss = epoch_loss

        if lr_decay != None:
            lr_decay.step(train_epoch_loss)

    # Finish training
    time_elapsed = time.time() - since
    print(' Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(' Best val loss: {:4f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    """
    When feature extract with pretrained model, we needn't retrain the parameters before FC
    But different when fine tune
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, feature_extract=False, use_pretrained=False, pre_model = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # Other wise we will need to define the structure by ourselves with forward function using module and sequential to organize
    
    model_ft = None
    input_size = 0
    output_size = 0

    if model_name == "UNet_Adapted":
        """ UNet_Adapted
        """
        model_ft = UNet_Adapted(n_Channels, n_Classes)
        if use_pretrained:
            model_ft.load_state_dict(torch.load(pre_model))
        input_size = 512
        output_size = 512
        learningrate = 1e-1

    elif model_name == "U_Net":
        """ U_Net
        """
        model_ft = U_Net(n_Channels, n_Classes)
        if use_pretrained:
            model_ft.load_state_dict(torch.load(pre_model))
        input_size = 572
        output_size = 388
        learningrate = 1e-1

    elif model_name == "FCN":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft = FCN(model_ft, n_Classes, 224)

        input_size = 224
        output_size = 224
        learningrate = 1e-3
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size, output_size, learningrate


if __name__ == "__main__":
    # Step1 Model:
    # Initialize the model for this run
    model_ft, input_size, output_size, learningrate = initialize_model(model_name, feature_extract, use_pretrained=use_pretrained, pre_model=pre_model_path)

    # Print the model we just instantiated
    # print(model_ft)

    # Step2 Dataset:
    # Data augmentation and normalization function for training
    # Just normalization for validation

    print(" >> Initializing Datasets and Dataloaders")

    # Create training and validation datasets
    image_datasets = {x: MaskDataset(os.path.join(data_dir, x), x, input_size, output_size) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers = 4) for x in ['train', 'val']}

    # Step3 Transfer to GPU
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Step4 Optimizer
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=learningrate) #optim.SGD(params_to_update, lr = 0.1, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='min',factor=0.1,patience=10) #optim.lr_scheduler.StepLR(optimizer_ft, step_size = 20, gamma=0.33)
    # Step5 Loss and train
    # Setup the loss fxn
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()
    print(' >> Model Created And Begin Training')
    # Train and evaluate
    model_ft, hist_val, hist_train = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=num_epochs)
    torch.save(model_ft.state_dict(), '../model/'+model_name+'_model_task2.pkl') #model = model_object.load_state_dict(torch.load('params.pkl'))

    # show result
    plt.figure(1)
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,num_epochs+1),hist_val,label = "validation")
    plt.plot(range(1,num_epochs+1),hist_train,label = "training")
    #plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()
