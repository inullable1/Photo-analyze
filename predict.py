from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolo11x.pt")

def print_model_classes():
    class_names = model.names
    print("\nДоступные классы в модели:")
    print("-" * 50)
    for class_id, class_name in class_names.items():
        print(f"ID: {class_id} -> {class_name}")
    print("-" * 50)

def predict_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Ошибка: Файл {image_path} не найден")
            return

        results = model(image_path)
        result = results[0]
        result.show()
        
        boxes = result.boxes
        if len(boxes) == 0:
            print("Объекты не обнаружены")
            return
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = result.names[int(class_id)]
            
            print(f"Обнаружен объект: {class_name}")
            print(f"Уверенность: {confidence:.2f}")
            print(f"Координаты: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            print("-" * 50)
            
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    print_model_classes()
    
    image_path = r"C:\Users\nullable\Desktop\solution\image copy 2.png"
    predict_image(image_path)
    input("Нажмите Enter для выхода...") 