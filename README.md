## 덕영고등학교 5팀 AI Project

도로 균열 및 포트홀, 싱크홀 감지 프로젝트


---

## 프로젝트 소개

yolov8 모델의 segmentation 학습을 통해 도로 균열 및 포트홀, 싱크홀을 감지

---

## 학습 코드 실행

⚠️ 직접 모델 학습시 종속성 문제로 인하여 docker 또는 colab을 사용할 것을 권장(직접 환경 설정시 [PyTorch 설치 가이드 참고](https://pytorch.org/get-started/locally/))


docker 이미지 다운로드 및 실행
```
# 약 30GB
docker pull ultralytics/ultralytics:latest

docker run -it --gpus all --ipc=host ultralytics:latest
```

코드 및 데이터셋 다운로드
```
cd ..
git clone https://github.com/gaeguli/yolov8_seg_CodeAndDatasets.git

cd yolov8_seg_CodeAndDatasets/datasets

rm datasets.zip
wget https://github.com/gaeguli/yolov8_seg_CodeAndDatasets/raw/master/datasets/datasets.zip
unzip datasets.zip
cd ..
python main.py
```
모델 파일 추출


학습 완료 후 외부 터미널에서 입력
```
docker cp <컨테이너_이름>:<학습이_왼료된_모델_파일경로> ./best.pt
```
