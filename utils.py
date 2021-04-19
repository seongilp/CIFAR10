from model.LeNet5_model import LeNet5_model
from model.ResNet_model import ResNet32_model
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageFolder

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def recall_model(cfg, save_path = None):
    if cfg.modelname == "LeNet5":
        model = LeNet5_model()
    elif cfg.modelname == "ResNet32":
        model = ResNet32_model()
    else:
        print("Wrong modelname.")
        quit()

    if save_path is not None:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, checkpoint

    return model

def imgshow(image, label, classes):
    print('========================================')
    print("The 1st image:")
    print(image)
    print('Shape of this image\t:', image.shape)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title('Label:%s' % classes[label])
    plt.show()
    print('Label of this image:', label, classes[label])

def evaluate(model, test_loader, device, verbose = False):
    correct_cnt = 0
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        pred = model.forward(img)
        _, top_pred = torch.topk(pred, k = 1, dim = -1)
        top_pred = top_pred.squeeze(dim=1)
        if verbose:
            print("--------------------------------------")
            print("truth:", classes[label])
            print("model prediction:", classes[top_pred])

        correct_cnt += int(torch.sum(top_pred == label))

    return correct_cnt

def train_data_load():
    # train data augmentation : 1) 데이터 좌우반전(2배). 2) size 4만큼 패딩 후 32의 크기로 random cropping
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10 dataset 다운로드
    train_data = dsets.CIFAR10(root='./dataset/', train=True, transform=transforms_train, download=True)
    val_data = dsets.CIFAR10(root="./dataset/", train=False, transform=transforms_val, download=True)

    return train_data, val_data

def eval_data_load():

    # CIFAR10 dataset 다운로드
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_data = dsets.CIFAR10(root='./dataset/', train=False, transform=transforms_test, download=True)
    return test_data

def test_data_load():
    transforms_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    imgs = ImageFolder('./test_example', transform=transforms_test)

    return imgs

def data_load(mode = "train"):
    if mode.lower() == "train":
        train_data, val_data = train_data_load()
        return train_data, val_data

    elif mode.lower() == "evaluation":
        test_data = eval_data_load()
        return test_data

    elif mode.lower() == "test example":
        test_data = test_data_load()
        return test_data
    
    else:
        print("Please write the correct mode [train, evaluation, test example]")
        exit()
