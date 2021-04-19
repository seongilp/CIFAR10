import torch
from torch.utils.data import DataLoader
from CIFAR10_configuration import Config
from model.ResNet_model import ResNet32_model
from utils import recall_model, data_load, evaluate

print('[CIFAR10_evaluation]')
cfg = Config()
classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

if __name__ == "__main__":
    
    #test_example에서 이미지 불러오기
    imgs = data_load(mode = "test example")
    test_loader = DataLoader(imgs, batch_size=1)

    # 저장된 state 불러오기
    #TODO
    #save_path 바꾸기
    save_path = "./saved_model/epoch_6.pth"
    #모델 불러오기
    model, checkpoint = recall_model(cfg, save_path)
    model.to(device)

    #평가
    evaluate(model, test_loader, device, verbose = True)

