import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from CIFAR10_configuration import Config
from torch.optim import lr_scheduler
from utils import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# configuration
cfg = Config()

print('[CIFAR10_training]')
print('Training with:', cfg.modelname)
# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def generate_batch(train_data, val_data):
    train_batch_loader = DataLoader(train_data, cfg.batch_size, shuffle=True)
    val_batch_loader = DataLoader(val_data, cfg.batch_size, shuffle=True)
    return train_batch_loader, val_batch_loader

def select_optimizer(cfg):
    if cfg.modelname == "LeNet5":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.modelname == "ResNet32":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        decay_epoch = [32000, 48000]
        step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=decay_epoch, gamma=0.1)

    return optimizer

def train_model(model, train_batch_loader, val_batch_loader, optimizer, criterion):
    # training 시작
    start_time = time.time()
    highest_val_acc = 0
    val_acc_list = []
    global_steps = 0
    epoch = 0
    print('========================================')
    print("Start training...")
    while True:
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for img, label in train_batch_loader:
            global_steps += 1
            # img.shape: [200,3,32,32]
            # label.shape: [200]

            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss

            train_batch_cnt += 1

            if global_steps >= cfg.finish_step:
                print("Training finished.")
                break

        ave_loss = train_loss / train_batch_cnt
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % ave_loss)
        print("training_time: %.2f minutes" % training_time)

        # validation (for early stopping)
        correct_cnt = evaluate(model, val_batch_loader, device, verbose = False)
        
        val_acc = correct_cnt / len(val_data) * 100
        print("validation dataset accuracy: %.2f" % val_acc)
        val_acc_list.append(val_acc)
        if val_acc > highest_val_acc:
            save_path = './saved_model/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)
            highest_val_acc = val_acc
        epoch += 1
        if global_steps >= cfg.finish_step:
            break

    epoch_list = [i for i in range(1, epoch + 1)]
    plt.title('Validation dataset accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, val_acc_list)
    plt.show()
    plt.savefig("./plot/result.png")


if __name__ == '__main__':
    # 데이터 로드
    # CIFAR10 dataset: [3,32,32] 사이즈의 이미지들을 가진 dataset
    train_data, val_data = data_load(mode = "train")

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # data 개수 확인
    print('The number of training data: ', len(train_data))
    print('The number of validation data: ', len(val_data))

    # shape 및 실제 데이터 확인
    image, label = train_data[1]
    imgshow(image, label, classes)

    # 학습 모델 생성
    model = recall_model(cfg)
    model.to(device)

    # 배치 생성
    train_batch_loader, val_batch_loader = generate_batch(train_data, val_data)

    #loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(cfg)

    #Train model
    train_model(model, train_batch_loader, val_batch_loader, optimizer, criterion)

