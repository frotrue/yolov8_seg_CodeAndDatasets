from ultralytics import YOLO

def main():
    model = YOLO('yolov8s-seg.pt')

    model.train(

        data='datasets/data.yaml',
        epochs=100,
        imgsz=640,
        batch=10,
        name='final',
        device='cuda',
        verbose=True
    )

    results = model.val()
    print(results)


if __name__ == '__main__':
    main()