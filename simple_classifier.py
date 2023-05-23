import numpy as np
# classifier model part
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob
import PIL
from PIL import Image
import shutil


# dataloader
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

# todo 여기는 GPU code입니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# todo 아마 여기만 참조해도 꽤 도움이 됩니다.
class Mean_classifier(nn.Module):
    def __init__(self, num_class):
        super(Mean_classifier, self).__init__()

        _res_net = torchvision.models.resnet50(pretrained=True)

        modules = list(_res_net.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(_res_net.fc.in_features, num_class)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

## code for deploy perpose
"""
    model_checkpoint_path = r'Z:\assistant\assistant_deploy\prometeus_v3_checkpoint_0.7229344844818115.pt'
    model = Image_Semantic_AwareClassifier_v3()
    model.load_state_dict(torch.load(model_checkpoint_path)['model_state_dict'])
    model.to(device)
    model.eval()
"""

# 여기는 학습용으로 만들고
# todo 차후에 deployment 관점에서 사용하기 위해서 파일별로 get하고 write하는 dataset을 만들어야 한다.

# trainer block
def train_loop(dataloader, num_class, model, loss_fn, optimizer, scaler):
    size = len(dataloader.dataset)
    for batch, (img, y) in enumerate(dataloader):
        # Compute prediction error
        y = torch.nn.functional.one_hot(y, num_classes=num_class).float()
        img, y = img.to(device), y.to(device)

        # Backpropagation
        optimizer.zero_grad()
        with autocast():
            pred = model(img)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test block 이후에 deploy block을 만들어서 실제 활용을 해야한다.
def test_loop(dataloader, model, loss_fn, result_path, epoch):
    size = len(dataloader.dataset)
    test_loss, correct, true_count, tot_true = 0.0, 0, 0, 0
    
    true_labels = []
    pred_labels = []

    
    with torch.no_grad():
        for img, y in dataloader:
            y = torch.nn.functional.one_hot(y, num_classes=2).float()
            img, y = img.to(device), y.to(device)

            pred = model(img)

            true_labels.extend(y.argmax(axis=1).cpu().numpy())
            pred_labels.extend(pred.argmax(axis=1).cpu().numpy())

            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred.argmax(axis=1) == y.argmax(axis=1)).type(torch.float).sum().item()
            true_count += torch.sum(((pred.argmax(axis=1) == y.argmax(axis=1)) * (pred.argmax(axis=1) == 1)) != 0)
            # tot_true += (pred.argmax(axis=1) == 1)
    test_loss /= size
    correct /= size
    
    fig, ax = plt.subplots()

    cm = confusion_matrix(true_labels, pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.savefig(os.path.join(result_path, f'confusion_matrix_{epoch}.png'))

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f},"
          f"true_count: {true_count / (size * 0.2)} \n")

    return (100 * correct), true_count / (size * 0.2)


# 여기서 CSV 로서 저장하고자 한다.
def deployment_loop(dataloader, model, result_path):
    size = len(dataloader.dataset)
    
    # 생각해보니 shuffle만 안하면 되는 거 아닌가?
    # 그래도 혹시나 모르니까
    # lamda_allocator = lambda x: dataset.dataset.dataset.imgs[x][0].split('/')[-1]
    # img_name_list = list(map(lamda_allocator, dataloader.dataset.indices))
    
    img_name_list = [x[0].split('/')[-1] for x in dataloader.dataset.imgs]
    pred_class_list = []
    pred_conf_list = []  # 해당 class의 confidence를 regress한다.
    
    with torch.no_grad():
        for batch_idx, (img, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = img.to(device)
            logit = model(img)
            # probability regression을 위해서 softmax를 넣어 준다.
            pred = F.softmax(logit, dim=1)
            pred_class, pred_conf = pred.max(axis=1)
            # pred_class = pred.argmax(axis=1)
            # pred_conf = pred[:, pred_class]
            pred_class_list.extend(pred_class.detach().cpu().numpy())
            pred_conf_list.extend(pred_conf.detach().cpu().numpy())

    with open(os.path.join(result_path, 'prediction.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_name', 'pred_class', 'pred_conf'])
        for row in tqdm(zip(img_name_list, pred_class_list, pred_conf_list), total=len(img_name_list)):
            writer.writerow(row)
    # 열 때는 읽어 드린 다음 dict로 해서 hash map하기

if __name__ == "__main__":
    # 여기수정
    result_path = r'/data/results/chronos/trial3'
    os.makedirs(result_path, exist_ok=True)
    random_seed = 777
    # 여기 수정
    dataset_path = r'/data/train/chronos'
    train_size = 0.8
    lr = 1e-5 # learning rate for fine tuning
    # batchsize는 최대한 크게 memory stall 없게
    train_batch_size = 16
    test_batch_size = 16
    train_model = True

    # todo 여기 수정
    num_class = 2
    
    #
    deployment_batch_size = 512
    model_checkpoint_path = r'/data/results/chronos/trial2/4_0.6777216792106628_85.00360490266763/checkpoint_0.6777216792106628.pt'
    deployment_dir = r'/data/train/deployments/RawImg_5011/Disaster'
    discard_dir = r'/data/train/corrupted_discarded_image_0511'
    os.makedirs(discard_dir, exist_ok=True)
    prediction_path = r'/data/results/chronos_5011_re'
    os.makedirs(prediction_path, exist_ok=True)
    
    if train_model:
        torch.manual_seed(random_seed)  # seed 고정
        model = Mean_classifier(num_class=num_class)  # todo 모델이 가정이 너무 이상할 수도 있으니까 vanilla version도 짠다.
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        rgb_transform = transforms.Compose([
            # resnet 맞게
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # todo 여기도 수정
        dataset = datasets.ImageFolder(dataset_path, transform=rgb_transform)
        train_split_generator = torch.Generator().manual_seed(random_seed)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_size * len(dataset)), len(dataset) - int(train_size * len(dataset))], generator=train_split_generator)

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, pin_memory=True, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4, pin_memory=False, shuffle=False)

        epochs = 100

        scaler = GradScaler()
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, num_class, model, loss_fn, optimizer, scaler)
            acc, rtpt = test_loop(test_dataloader, model, loss_fn, result_path, epoch=t)
            # model checkpointer
            checkpoint = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            # confusion matrix로 사용 가능성 검증한다.
            rtpt = rtpt.detach().cpu().item()
            os.makedirs(os.path.join(result_path, str(t) + '_' + str(rtpt) + '_' + str(acc)), exist_ok=True)
            torch.save(checkpoint, os.path.join(os.path.join(result_path, str(t) + '_' + str(rtpt) + '_' + str(acc)),
                                                f'checkpoint_{rtpt}.pt'))

        print("Done!")

    else:
        rgb_transform = transforms.Compose([
            # resnet 맞게
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        deployment_dataset = datasets.ImageFolder(deployment_dir, transform=rgb_transform)

        deployment_loader = DataLoader(deployment_dataset, batch_size=deployment_batch_size, num_workers=4, pin_memory=False, shuffle=False)
        
        model = Mean_classifier() 
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()
        
        deployment_loop(deployment_loader, model, prediction_path)
        print('prediction done')
