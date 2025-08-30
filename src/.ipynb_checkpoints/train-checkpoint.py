from ultralytics import YOLO

def train_model():
    """
    Trains a YOLOv8 model on the custom dental dataset.
    """
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        project='runs/detect',
        name='dental_experiment_1',
        exist_ok=True 
    )
    
    print("Training complete.")
    print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_model()