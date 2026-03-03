import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import random
import pandas as pd
import re


### 모델 평가 후 결과를 CSV 파일로 저장하는 함수 추가
def save_predictions_to_csv(image_paths, preds, labels, output_file):

    df = pd.DataFrame({
        'Image Path': image_paths,  # 이미지 경로
        'Actual Label': labels,     # 실제 레이블 값
        'Predicted Label': preds    # 예측된 레이블 값
    })

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")



def get_image_list(folder):
    """폴더 내 이미지 파일 경로 전부 가져오기"""
    return sorted(glob(os.path.join(folder, '*.jpg')) + glob(os.path.join(folder, '*.png')))


def normalize_filename(fname):
    # 🔧 "N일차 수확" 또는 "일차 수확" 패턴 제거
    # 예: butterhead_2_1_240924_15_5일차 수확_ins_0.jpg → butterhead_2_1_240924_15_ins_0.jpg
    fname = re.sub(r'_\d*일차\s*수확', '', fname)  # "_5일차 수확" 제거
    fname = re.sub(r'_일차\s*수확', '', fname)     # "_일차 수확" 제거
    fname = re.sub(r'_\d+일차_', '_', fname)       # "_14일차_" → "_"
    fname = re.sub(r'_\d+일차', '', fname)         # "_14일차" → ""
    fname = re.sub(r'_일차', '', fname)            # "_일차" → ""
    return fname



def data_split(base_dir, weight_csv=None, Y_column=None):
    """
    이미지 폴더 구조 기반 데이터 분할 로더
    (ex: pre-test-data/Image-FastSAM/split/seed_42/butterhead/1차재배)

    - train, val, test 하위 폴더에서 이미지 경로 자동 로드
    - weight_csv가 있으면, 레이블을 CSV에서 매칭
    - weight_csv가 없으면, 더미 레이블(0)로 대체
    """

    # 하위 폴더 경로 정의
    train_dir = os.path.join(base_dir, 'train')
    val_dir   = os.path.join(base_dir, 'val')
    test_dir  = os.path.join(base_dir, 'test')

    train_images = get_image_list(train_dir)
    val_images   = get_image_list(val_dir)
    test_images  = get_image_list(test_dir)

    # CSV에서 weight 매칭 (선택적)
    if weight_csv and os.path.exists(weight_csv):
        df = pd.read_csv(weight_csv)
        df['img_path'] = df['img_path'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
        Y_map = dict(zip(df['img_path'], df[Y_column]))

        # def get_labels(img_paths):
        #     labels = []
        #     for path in img_paths:
        #         fname = os.path.basename(path)
        #         if fname not in Y_map:
        #             raise KeyError(f"⚠️ '{fname}' not found in weight CSV file: {weight_csv}")
        #         labels.append(Y_map[fname])
        #     return labels


        def get_labels(img_paths):
            labels = []
            valid_paths = []
            for path in img_paths:
                fname = os.path.basename(path)
                norm_name = normalize_filename(fname)  # 파일명 정규화

                if norm_name not in Y_map:
                    print(f"⚠️ '{fname}' → '{norm_name}' (정규화 후에도 CSV에 없음, 건너뜀)")
                    continue

                labels.append(Y_map[norm_name])
                valid_paths.append(path)
            return valid_paths, labels
        

        train_images, train_labels = get_labels(train_images)
        val_images, val_labels     = get_labels(val_images)
        test_images, test_labels   = get_labels(test_images)

    else: # CSV 없을 경우
        raise FileNotFoundError(
            f"❌ weight CSV 파일이 존재하지 않거나 경로가 잘못되었습니다.\n"
            f"  확인된 경로: {weight_csv}"
        )

    return train_images, val_images, test_images, train_labels, val_labels, test_labels



# ==============================
# CustomDataset 수정 버전
# ==============================
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        - image_paths: 실제 이미지 경로 리스트
        - labels: 해당 이미지의 weight 값 (또는 더미)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
