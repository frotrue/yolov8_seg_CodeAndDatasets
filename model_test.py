from ultralytics import YOLO
import cv2

model = YOLO('/home/user/myenv/YoloV8_test/runs/segment/final7/weights/best.pt')  # 학습된 모델 경로

img = cv2.imread('/home/user/myenv/YoloV8_test/test_data_img/20150108_lotte6.jpg')  # 추론할 이미지

results = model(img) # 필요시 conf 옵션 넣기

# 결과 안에 클래스, 좌표 등 포함
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"Detected {label} with confidence {conf:.2f}")

# 결과 시각화
results[0].show()
# 또는 저장
results[0].save(filename='output.jpg')