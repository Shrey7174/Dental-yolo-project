
from ultralytics import YOLO
import os
import glob
import cv2  
from post_process import apply_anatomical_rules

def evaluate_and_predict_with_post_processing():
    
    MODEL_PATH = 'runs/detect/dental_experiment_1/weights/best.pt'
    TEST_IMAGES_DIR = 'data/images/test/'
    RESULTS_DIR = 'results/post_processed/'

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at '{MODEL_PATH}'. Please check the path.")
        return

    print("Loading the trained model...")
    model = YOLO(MODEL_PATH)

    test_image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))

    if not test_image_paths:
        print(f"No test images found in '{TEST_IMAGES_DIR}'.")
        return

    print(f"\nFound {len(test_image_paths)} test images. Running predictions...")
    raw_results = model.predict(source=test_image_paths, conf=0.5)

    print("\nApplying post-processing anatomical rules...")
    for i, result in enumerate(raw_results):
        image_path = test_image_paths[i]
        img = cv2.imread(image_path)
        
        clean_boxes = apply_anatomical_rules(result.boxes, result.orig_shape[0], model.names)

        for box in clean_boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            label = f"{model.names[class_id]} {confidence:.2f}"

            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        output_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
        cv2.imwrite(output_path, img)

    print(f"\nProcess complete. Post-processed images saved in '{RESULTS_DIR}'.")

if __name__ == '__main__':
    evaluate_and_predict_with_post_processing()