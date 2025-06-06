import os
import random
import shutil

# 경로 설정
img_dir = "datasets/before_data/images"
label_dir = "datasets/before_data/labels"
output_base = "datasets/after_data"

# 비율 설정
train_ratio = 0.8

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]

# 셔플 후 split
random.shuffle(image_files)
split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 디렉토리 생성
for folder in ["train", "val"]:
    os.makedirs(os.path.join(output_base, "images", folder), exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels", folder), exist_ok=True)

# 파일 복사
def move_files(file_list, split):
    for f in file_list:
        name = os.path.splitext(f)[0]
        # 이미지 복사
        shutil.copy(os.path.join(img_dir, f), os.path.join(output_base, "images", split, f))
        # 라벨 복사
        label_path = os.path.join(label_dir, f"{name}.txt")
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_base, "labels", split, f"{name}.txt"))
        else:
            print(f"라벨 파일 누락: {label_path}")

move_files(train_files, "train")
move_files(val_files, "val")

print(f"총 {len(train_files)}개 train, {len(val_files)}개 val 파일로 나눴습니다.")
