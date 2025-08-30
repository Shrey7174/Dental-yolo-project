from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='data.yaml',
        epochs=200,  
        imgsz=640,
        project='runs/detect',
        name='dental_experiment_2',  
        exist_ok=True,

        
        degrees=10,      
        translate=0.1,  
        scale=0.1,     
        shear=5,         
        flipud=0.1,      
        fliplr=0.5       
    )

    print("Training complete.")

if __name__ == '__main__':
    train_model()