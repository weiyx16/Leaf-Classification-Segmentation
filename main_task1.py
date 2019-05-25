'''

Leaf Classification with finetuning or feature extraction on pretrained resnet-50
3 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# hyperparameter
# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
# structure in classification:
# data_dir/train/Leaf_0/xxx.png
# data_dir/train/Leaf_1/xy.png
# data_dir/train/Leaf_2/xxz.png

# data_dir/val/Leaf_0/lxx.png
# data_dir/val/Leaf_1/gy.png
# data_dir/val/Leaf_2/llz.png
data_dir = "../data/Task1"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def train_model(model, dataloaders, criterion, optimizer, num_epochs=15, is_inception=False):
    since = time.time()

    # validation accuracy
    val_acc_history = []
    # for save the best accurate model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            # Another way: dataIter = iter(dataloaders[phase]) then next(dataIter)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # criterion define the loss function
                        # calculate the loss also on the validation set
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    # along the batch axis
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize parameters only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(' Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(' Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    """
    When feature extract with pretrained model, we needn't retrain the parameters before FC
    But different when fine tune
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # Other wise we will need to define the structure by ourselves with forward function using module and sequential to organize
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet52
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        model_ft.fc = nn.Linear(num_ftrs, num_classes) # replace fc with 2048 to num_class
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == "__main__":
    # Step1 Model:
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Step2 Dataset:
    # Data augmentation and normalization function for training
    # Also rgb2gray
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(" >> Initializing Datasets and Dataloaders")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

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
    
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        params_to_update = model_ft.parameters()
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                # print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=1e-3) #optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Step5 Loss and train
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    print(' >> Model Created And Begin Training')
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    torch.save(model_ft.state_dict(), '../model/model_task1.pkl') #model = model_object.load_state_dict(torch.load('params.pkl'))
    # show result
    plt.figure(1)
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),hist)
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    #plt.legend()
    plt.show()

    # result example
    # [*] Epoch 0/9
    # [**] Begin train ...
    # train Loss: 0.4639 Acc: 0.8664
    # [**] Begin val ...
    # val Loss: 0.1465 Acc: 0.9859
    # [*] Epoch 1/9
    # [**] Begin train ...
    # train Loss: 0.1661 Acc: 0.9724
    # [**] Begin val ...
    # val Loss: 0.0821 Acc: 0.9929
    # [*] Epoch 2/9
    # [**] Begin train ...
    # train Loss: 0.1250 Acc: 0.9747
    # [**] Begin val ...
    # val Loss: 0.0569 Acc: 0.9965
    # [*] Epoch 3/9
    # [**] Begin train ...
    # train Loss: 0.1146 Acc: 0.9675
    # [**] Begin val ...
    # val Loss: 0.0564 Acc: 0.9823
    # [*] Epoch 4/9
    # [**] Begin train ...
    # train Loss: 0.0954 Acc: 0.9767
    # [**] Begin val ...
    # val Loss: 0.0415 Acc: 0.9965
    # [*] Epoch 5/9
    # [**] Begin train ...
    # train Loss: 0.0799 Acc: 0.9782
    # [**] Begin val ...
    # val Loss: 0.0333 Acc: 0.9965
    # [*] Epoch 6/9
    # [**] Begin train ...
    # train Loss: 0.0652 Acc: 0.9847
    # [**] Begin val ...
    # val Loss: 0.0345 Acc: 0.9965
    # [*] Epoch 7/9
    # [**] Begin train ...
    # train Loss: 0.0698 Acc: 0.9824
    # [**] Begin val ...
    # val Loss: 0.0285 Acc: 1.0000
    # [*] Epoch 8/9
    # [**] Begin train ...
    # train Loss: 0.0579 Acc: 0.9885
    # [**] Begin val ...
    # val Loss: 0.0327 Acc: 0.9965
    # [*] Epoch 9/9
    # [**] Begin train ...
    # train Loss: 0.0597 Acc: 0.9839
    # [**] Begin val ...
    # val Loss: 0.0244 Acc: 0.9929

    # Training complete in 32m 53s
    # Best val Acc: 1.000000
