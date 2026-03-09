# PyTorch 관련 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# 데이터 및 분석 관련 라이브러리
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import datetime
from datetime import datetime
import pytz


import matplotlib.pyplot as plt
plt.ioff()
import os
from tqdm import tqdm

# 한국 시간 (KST) 설정
kst = pytz.timezone('Asia/Seoul')
start_time = datetime.now(kst)


# from trainer import write_log, train_model, evaluate_model
# from utils.data_processing import CustomDataset, data_split, get_augmented_imgs, get_augmented_image_paths, assign_labels_to_augmented_images, save_predictions_to_csv
from utils.datasets import CustomDataset, data_split, save_predictions_to_csv
from utils.models import get_model


import argparse

def get_args():
    parser = argparse.ArgumentParser()

    ## 작물 선택
    parser.add_argument('--plant', type=str, default='butterhead', help='Type of plant ex) butterhead')                  
    parser.add_argument('--order1', type=int, default='2', help='몇 차 재배')  
    ## 테스크 선택
    parser.add_argument('--Y_column', type=str, default='weight_last', choices=['day', 'weight_day', 'weight_last'],  
                            help="예측 테스크: 생육일자(day), 현재중량(weight_day), 최종중량(weight_last)")
    parser.add_argument('--matching_CSV', type=str, default='/workspace/2024_CJ/3_DAP_prediction/pretest-matching-fastsam.csv', help='몇 차 재배')  
    
    ## 증강 배수 선택
    parser.add_argument('--augmentation_type', default=0, type=int, help='몇배증강')  

    ## 실험 세팅
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU index to use (default: 0)')  # 사용할 GPU 번호 설정
    parser.add_argument('--model_name', type=str, default='efficientnet_b2', 
                            choices=['resnet18', 'vgg16_bn', 'efficientnet_b2', 'densenet121'],
                            help='학습할 모델의 이름 (resnet18, vgg16_bn, efficientnet_b2, densenet121)')    
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005, help='학습률 (learning rate)')
    parser.add_argument('--step_size', type=int, default=70, help='LR 스케줄러의 step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR 스케줄러의 감마 값')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

args = get_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# train_dir = f"pre-test-data/Image-FastSAM/split/seed_{args.seed}/{args.plant}/{args.order1}차재배/train"
# val_dir = f"pre-test-data/Image-FastSAM/split/seed_{args.seed}/{args.plant}/{args.order1}차재배/val"
# test_dir = f"pre-test-data/Image-FastSAM/split/seed_{args.seed}/{args.plant}/{args.order1}차재배/test"

# # [FIXME]
# aug_dir1 = f"pre-test-data/Image-SAM-aug/{args.plant}_{args.order1}-{order2_1}_train_blur_color_12x/bbox/img"
# aug_dir2 = f"pre-test-data/Image-SAM-aug/{args.plant}_{args.order1}-{order2_2}_train_blur_color_12x/bbox/img"
# aug_dir3 = f"pre-test-data/Image-SAM-aug/{args.plant}_{args.order1}-{args.order2_test}_train_blur_color_12x/bbox/img"


# 학습된 가중치 저장 경로
train_weight_dir = f'PRETEST/weight/{args.model_name}/seed_{args.seed}/{args.Y_column}/{args.plant}_{args.order1}/AUG_{args.augmentation_type}/{args.model_name}'
os.makedirs(train_weight_dir, exist_ok=True)
train_weight_save_path = f'{train_weight_dir}/best_model.pth'

# 예측 결과 저장 경로
csv_dir = f'PRETEST/results/{args.model_name}/seed_{args.seed}/{args.Y_column}/{args.plant}_{args.order1}/AUG_{args.augmentation_type}/{args.model_name}'
os.makedirs(csv_dir, exist_ok=True)
predictions_csv = f'{csv_dir}/results.csv'

# 기존의 log_file_path 정의
log_dir = f'PRETEST/log/{args.model_name}/seed_{args.seed}/{args.Y_column}/{args.plant}_{args.order1}/AUG_{args.augmentation_type}/{args.model_name}'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f'train.log')
log_file_path_mae_mape = os.path.join(log_dir, f'MAE_MAPE.log')

###########################################        log        ####################################################################

def write_log(message, log_file_path, print_to_terminal=True):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    if print_to_terminal:
        print(message)



# 일반 로그 기록
write_log(f"Code start time: {start_time}", log_file_path)
write_log(f"SEED: {args.seed}", log_file_path)
write_log(f"Plant: {args.plant}", log_file_path)
write_log(f"testing data: {args.order1}", log_file_path)
write_log(f"선택된 모델: {args.model_name}", log_file_path)
write_log(f"학습률: {args.lr}", log_file_path)
write_log(f"Step Size: {args.step_size}, Gamma: {args.gamma}, Batch Size: {args.batch_size}", log_file_path)
write_log(f"색상 증강: {args.augmentation_type} 배 ", log_file_path)

# MAE MAPE 로그 기록
write_log(f"Plant: {args.plant}", log_file_path_mae_mape, print_to_terminal=False)
write_log(f"SEED: {args.seed}", log_file_path_mae_mape, print_to_terminal=False)
write_log(f"testing data: {args.order1}", log_file_path_mae_mape, print_to_terminal=False)
write_log(f"선택된 모델: {args.model_name}", log_file_path_mae_mape, print_to_terminal=False)
write_log(f"색상 증강: {args.augmentation_type} 배 ", log_file_path_mae_mape, print_to_terminal=False)


########################################################################################################################################


### 모델 학습 
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_loss = np.inf
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        write_log(f'Epoch {epoch+1}/{num_epochs}', log_file_path)
        write_log('-' * 10, log_file_path)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # 프로그래스바
            phase_size = dataset_sizes[phase]
            with tqdm(total=phase_size, desc=f'Epoch {epoch+1}/{num_epochs} [{phase}]', unit='img') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()  

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).squeeze() 
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    pbar.update(inputs.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = model.state_dict()
                    torch.save(best_model, train_weight_save_path)
                    write_log('Best Model saved.', log_file_path)

            write_log(f'{phase} Loss: {epoch_loss:.4f}', log_file_path)



    ### 학습 손실 및 검증 손실 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    return model



### 모델 평가
def evaluate_model(model, dataloaders):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            outputs = model(inputs).squeeze()
            all_preds.extend(outputs.unsqueeze(dim=0).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels



###########################################           main        ####################################################################
import random
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


### 데이터 전처리 변환 정의
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((540, 960)), # [FIXME]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((540, 960)), # [FIXME]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


weight_csv = args.matching_CSV
base_dir = f"pre-test-data/Image-FastSAM/split/seed_{args.seed}/{args.plant}/{args.order1}차재배"
train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = data_split(base_dir, weight_csv, Y_column=args.Y_column)

# 데이터셋 생성
train_dataset = CustomDataset(train_image_paths, train_labels, transform=data_transforms['train'])
val_dataset   = CustomDataset(val_image_paths, val_labels, transform=data_transforms['val'])
test_dataset  = CustomDataset(test_image_paths, test_labels, transform=data_transforms['val'])


# [FIXME]
# # 증강된 이미지 합치기


# 데이터 로더 설정
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6),  
    'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6),
    'test': DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)  
}


dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}


#### 데이터 개수
write_log(f'Load images ...', log_file_path)
write_log(f' - Number of Train: {dataset_sizes["train"]}', log_file_path)
write_log(f' - Number of Validation: {dataset_sizes["val"]}', log_file_path)
write_log(f' - Number of Test: {dataset_sizes["test"]}',log_file_path)



###  모델 불러오기     
model = get_model(args.model_name)
model = model.to(device)

### 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)



## 모델 학습
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.num_epochs)

### 모델 평가
model_ft.load_state_dict(torch.load(train_weight_save_path))
preds, labels = evaluate_model(model_ft, dataloaders)

# 평가지표
write_log(f"Test MAE: {mean_absolute_error(preds, labels):.4f}", log_file_path)
write_log(f"Test MAPE: {mean_absolute_percentage_error(labels, preds):.6f}",log_file_path )

write_log(f"Test MAE: {mean_absolute_error(preds, labels):.4f}", log_file_path_mae_mape,print_to_terminal=False)
write_log(f"Test MAPE: {mean_absolute_percentage_error(labels, preds):.6f}",log_file_path_mae_mape ,print_to_terminal=False)
write_log(f"-------------------------------------------------------------------", log_file_path_mae_mape,print_to_terminal=False)

save_predictions_to_csv(test_image_paths, preds, labels, output_file=predictions_csv)


end_time = datetime.now(kst)
write_log(f"Code end time: {end_time}", log_file_path)
write_log(f"Total duration: {end_time - start_time}", log_file_path)