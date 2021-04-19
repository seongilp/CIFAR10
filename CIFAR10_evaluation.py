import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CIFAR10_configuration import Config
from model.LeNet5_model import LeNet5_model
from model.ResNet_model import ResNet32_model
from utils import *

print('[CIFAR10_evaluation]')
cfg = Config()
# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def generate_batch(test_data):
    test_batch_loader = DataLoader(test_data, cfg.batch_size, shuffle=True)
    return test_batch_loader

if __name__ == "__main__":

    # 데이터 로드
    test_data = data_load()
    test_data = data_load(mode = "evaluation")
    # data 개수 확인
    print('The number of test data: ', len(test_data))
    # 배치 생성
    test_batch_loader = generate_batch(test_data)

    # test 시작
    acc_list = []

    #TODO
    #save_path 수정하기
    save_path = "./saved_model/epoch_131.pth"

    # 저장된 model 불러오기
    model, checkpoint = recall_model(cfg, save_path = save_path)
    model.to(device)

    correct_cnt = evaluate(model, test_batch_loader, device, verbose = False)

    accuracy = correct_cnt / len(test_data) * 100
    print("accuracy of the %d epoch trained model : %.2f%%"%(checkpoint["epoch"], accuracy))
    acc_list.append(accuracy)


