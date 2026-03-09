# inference.py

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

# 시간 및 날짜 관련 라이브러리
import datetime
from datetime import datetime
import pytz

# 시각화 라이브러리
import matplotlib.pyplot as plt
plt.ioff()

# 기타 유틸리티
import os
from tqdm import tqdm

# 한국 시간 (KST) 설정
kst = pytz.timezone('Asia/Seoul')
start_time = datetime.now(kst)


# from trainer import write_log, train_model, evaluate_model
from utils.data_processing import CustomDataset, data_split, save_predictions_to_csv
from utils.models import get_model


import argparse
def get_args():
    parser = argparse.ArgumentParser()

    ## 작물 선택
    parser.add_argument('--plant', type=str, default='butterhead', help='Type of plant ex) butterhead')                  
    parser.add_argument('--order1', type=str, default='2', help='몇 차 재배')  
    parser.add_argument('--order2_test', type=int, default=1, help='Which order2 value to use for testing (1, 2, or 3)')  


    ## 테스크 선택
    parser.add_argument('--Y_column', type=str, default='weight_last', choices=['day', 'weight_day', 'weight_last'],  
                            help="예측 테스크: 생육일자(day), 현재중량(weight_day), 최종중량(weight_last)")
    ## 증강 배수 선택
    parser.add_argument('--augmentation_type', default=0, type=int, help='몇배증강')  

    ## 실험 세팅
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU index to use (default: 0)')  # 사용할 GPU 번호 설정
    parser.add_argument('--model_name', type=str, default='resnet18', 
                            choices=['resnet18', 'vgg16_bn', 'efficientnet_b2', 'densenet121'],
                            help='학습할 모델의 이름 (resnet18, vgg16_bn, efficientnet_b2, densenet121)')    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005, help='학습률 (learning rate)')
    parser.add_argument('--step_size', type=int, default=70, help='LR 스케줄러의 step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR 스케줄러의 감마 값')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2028)
    
    return parser.parse_args()
args = get_args()

# 통합판 설정 
if args.order2_test == 1:
    order2_1 = '2'
    order2_2 = '3'
elif args.order2_test == 2:
    order2_1 = '1'
    order2_2 = '3'
elif args.order2_test == 3:
    order2_1 = '1'
    order2_2 = '2'

print(f"args.order2_test: {args.order2_test}, type: {type(args.order2_test)}")

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 파일 경로
test_dir = f"pre-test-data/Image-SAM/{args.plant}_{args.order1}-{args.order2_test}_test/bbox/img"

# 로그 경로
log_dir = f'logs_{args.Y_column}'
os.makedirs(log_dir, exist_ok=True)

# 학습된 가중치 저장 경로
os.makedirs('weight', exist_ok=True)
train_weight_save_path = f'weight/{args.Y_column}_{args.plant}_{args.order1}_PRETEST_augmentation_color_{args.augmentation_type}_{args.model_name}.pth'

# 예측 결과 저장 경로
os.makedirs('results', exist_ok=True)
predictions_csv = f'results/{args.Y_column}_{args.plant}_{args.order2_test}_augmentation_color_{args.augmentation_type}_{args.model_name}.csv'

###########################################        log        ####################################################################

def write_log(message, log_file_path, print_to_terminal=True):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    if print_to_terminal:
        print(message)

# 기존의 log_file_path 정의
log_file_path = os.path.join(log_dir, f'{args.plant}_{args.order2_test}_{args.model_name}_augmentation_{args.augmentation_type}_{args.Y_column}.log')
log_file_path_mae_mape = os.path.join(log_dir, f'MAE_MAPE_{args.plant}_{args.order2_test}.log')

# 일반 로그 기록
write_log(f"Code start time: {start_time}", log_file_path)
write_log(f"Plant: {args.plant}", log_file_path)
write_log(f"testing data: {args.order2_test}", log_file_path)
write_log(f"선택된 모델: {args.model_name}", log_file_path)
write_log(f"학습률: {args.lr}", log_file_path)
write_log(f"Step Size: {args.step_size}, Gamma: {args.gamma}, Batch Size: {args.batch_size}", log_file_path)
write_log(f"색상 증강: {args.augmentation_type} 배 ", log_file_path)

# MAE MAPE 로그 기록
write_log(f"Plant: {args.plant}", log_file_path_mae_mape,print_to_terminal=False)
write_log(f"testing data: {args.order2_test}", log_file_path_mae_mape,print_to_terminal=False)
write_log(f"선택된 모델: {args.model_name}", log_file_path_mae_mape,print_to_terminal=False)
write_log(f"색상 증강: {args.augmentation_type} 배 ", log_file_path_mae_mape, print_to_terminal=False)


#######################################################################################################################################
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

### 데이터 전처리 변환 정의
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((960, 540)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


## 테스트 데이터셋
_, _, test_image_paths, _, _, test_labels = data_split(f"weight_info/{args.Y_column}/{args.plant}_{args.order1}-{args.order2_test}.csv", args.Y_column)
test_dataset = CustomDataset(test_image_paths, test_labels, test_dir, transform=data_transforms['val'])


# 데이터 로더 설정
dataloaders = {'test': DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)  }
dataset_sizes = {'test': len(test_dataset)}

#### 데이터 개수
write_log(f'Load images ...', log_file_path)
write_log(f' - Number of Test: {dataset_sizes["test"]}',log_file_path)

###  모델 불러오기     
model = get_model(args.model_name)
model = model.to(device)

### 모델 평가
model.load_state_dict(torch.load(train_weight_save_path))
preds, labels = evaluate_model(model, dataloaders)

######################################################################################################################

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