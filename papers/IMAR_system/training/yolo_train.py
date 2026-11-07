from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    model.train(data='dataset_1/data.yaml', epochs=100, imgsz=1024, batch = 8)