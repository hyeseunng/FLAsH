import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


from datetime import datetime
from sklearn.model_selection import train_test_split




def data_split(csv_file, weight_column):
    """
    데이터 분할 (test: 하루 중 가운데 시간, train&val: 나머지 시간대)
    """
    # CSV 파일 불러오기
    data = pd.read_csv(csv_file)

    # 'date_hour' 열을 날짜와 시간으로 분리
    data['date'] = data['date_hour'].apply(lambda x: x.split()[0])  # 날짜 추출
    data['hour'] = data['date_hour'].apply(lambda x: int(x.split()[1].split(':')[0]))  # 시간(정수) 추출

    # 이미지 경로 생성
    data['img_path'] = data['img_path'].apply(lambda x: x + '.png')

    train_val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # 날짜별로 그룹화하여 데이터 분리
    for date, group in data.groupby('date'):
        # 시간대 기준으로 정렬
        group = group.sort_values(by='hour')

        # 시간대가 3개일 경우 처리
        unique_hours = group['hour'].unique()
        if len(unique_hours) == 3:
            # 중간 시간대를 test로 사용
            middle_hour = unique_hours[1]  # 중간 시간대 추출
            test = group[group['hour'] == middle_hour]  # 중간 시간대에 해당하는 데이터 추출
            train_val = group[group['hour'] != middle_hour]  # 나머지 시간대는 train/val로 사용

            # test 데이터와 train/val 데이터를 각각 모음
            test_data = pd.concat([test_data, test])
            train_val_data = pd.concat([train_val_data, train_val])

    # train/val 데이터를 9:1로 분할
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

    # 이미지 경로 및 레이블을 각각 리스트로 추출
    train_image_paths = train_data['img_path'].tolist()
    val_image_paths = val_data['img_path'].tolist()
    test_image_paths = test_data['img_path'].tolist()

    train_labels = train_data[weight_column].tolist()
    val_labels = val_data[weight_column].tolist()
    test_labels = test_data[weight_column].tolist()

    return train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels







# 증강 기법에 따라 이미지를 필터링하는 함수
def get_augmented_imgs(base_dir, aug_type):
    """
    주어진 디렉토리에서 증강 유형(aug_type)에 따라 이미지를 필터링합니다.
    
    :param base_dir: 이미지가 저장된 디렉토리 경로
    :param aug_type: 증강 유형 (1 ~ 21배)
    :return: 증강 유형에 맞는 이미지 파일 경로 리스트
    """
    # 폴더 내 모든 이미지 파일 이름을 가져옴
    images = [img for img in os.listdir(base_dir) if img.endswith('.png')]

    if aug_type == 21:
        # 모든 이미지 사용
        aug_img_paths = images
    else:
        # aug_type이 3, 6, 9, 12, 15, 18 중 하나일 경우
        valid_indices = list(range(1, aug_type + 1))
        aug_img_paths = [img for img in images if int(img.split('_')[-1].split('.')[0]) in valid_indices]

    # 선택된 이미지 수 출력
    num_imgs = len(aug_img_paths)
    print(f"Number of selected images: {num_imgs}")

    return aug_img_paths






def get_augmented_image_paths(image_paths, base_dir, aug_type):
    augmented_image_paths = []

    # 'base_dir'에서 증강된 이미지 파일을 가져옴
    all_augmented_images = get_augmented_imgs(base_dir, aug_type)

    for img_path in image_paths:
        # 원본 이미지 이름에서 확장자를 제거하고 '_'로 끝나는 모든 증강 이미지 추가
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # 증강 이미지들 중 해당 이미지 이름으로 시작하는 이미지들을 필터링하여 이름만 저장
        related_aug_images = [img for img in all_augmented_images if img.startswith(img_name + "_")]
        augmented_image_paths.extend(related_aug_images)

    return augmented_image_paths


def assign_labels_to_augmented_images(train_image_paths, train_labels, train_aug_img_paths):
    # 원본 이미지 이름과 레이블을 딕셔너리 형태로 저장
    label_dict = {os.path.splitext(img_path)[0]: label for img_path, label in zip(train_image_paths, train_labels)}

    # 증강 이미지에 해당하는 레이블을 저장할 리스트
    train_aug_labels = []

    # 각 증강 이미지에 대해 원본 이미지 이름 추출 후, 해당하는 레이블 찾기
    for aug_img_path in train_aug_img_paths:
        # 증강 이미지 이름에서 _X 부분을 제거하여 원본 이미지 이름 추출
        original_img_name = "_".join(os.path.basename(aug_img_path).split('_')[:-1])

        # 딕셔너리에서 원본 이미지 이름에 해당하는 레이블을 찾아 추가
        label = label_dict.get(original_img_name)
        if label is not None:
            train_aug_labels.append(label)
        else:
            print(f"Warning: No label found for {original_img_name}")

    return train_aug_labels





class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, image_dir, transform=None):
        # image_paths에 구체적인 디렉토리 경로 추가
        self.image_paths = [os.path.join(image_dir, img_path) for img_path in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 경로에서 이미지 불러오기
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, label
    


### 모델 평가 후 결과를 CSV 파일로 저장하는 함수 추가
def save_predictions_to_csv(image_paths, preds, labels, output_file):

    df = pd.DataFrame({
        'Image Path': image_paths,  # 이미지 경로
        'Actual Label': labels,     # 실제 레이블 값
        'Predicted Label': preds    # 예측된 레이블 값
    })

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")



#### 로그 파일 관리
import os

def write_log(message, log_file_path, print_to_terminal=True):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    if print_to_terminal:
        print(message)
