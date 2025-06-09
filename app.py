from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import logging
import traceback
from werkzeug.utils import secure_filename
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import shutil
from functools import partial

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Исправление загрузки весов PyTorch
original_torch_load = torch.load
torch.load = partial(original_torch_load, weights_only=False)
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# Словарь для перевода классов
CLASS_TRANSLATIONS = {
    0: "человек",
    1: "велосипед",
    2: "автомобиль",
    3: "мотоцикл",
    4: "самолет",
    5: "автобус",
    6: "поезд",
    7: "грузовик",
    8: "лодка",
    9: "светофор",
    10: "пожарный гидрант",
    11: "знак остановки",
    12: "парковочный счетчик",
    13: "скамейка",
    14: "птица",
    15: "кошка",
    16: "собака",
    17: "лошадь",
    18: "овца",
    19: "корова",
    20: "слон",
    21: "медведь",
    22: "зебра",
    23: "жираф",
    24: "рюкзак",
    25: "зонт",
    26: "сумка",
    27: "галстук",
    28: "чемодан",
    29: "фрисби",
    30: "лыжи",
    31: "сноуборд",
    32: "спортивный мяч",
    33: "воздушный змей",
    34: "бейсбольная бита",
    35: "бейсбольная перчатка",
    36: "скейтборд",
    37: "серфборд",
    38: "теннисная ракетка",
    39: "бутылка",
    40: "бокал для вина",
    41: "чашка",
    42: "вилка",
    43: "нож",
    44: "ложка",
    45: "миска",
    46: "банан",
    47: "яблоко",
    48: "сэндвич",
    49: "апельсин",
    50: "брокколи",
    51: "морковь",
    52: "хот-дог",
    53: "пицца",
    54: "пончик",
    55: "торт",
    56: "стул",
    57: "диван",
    58: "комнатное растение",
    59: "кровать",
    60: "обеденный стол",
    61: "туалет",
    62: "монитор",
    63: "ноутбук",
    64: "мышь",
    65: "пульт",
    66: "клавиатура",
    67: "мобильный телефон",
    68: "микроволновая печь",
    69: "духовая печь",
    70: "тостер",
    71: "раковина",
    72: "холодильник",
    73: "книга",
    74: "часы",
    75: "ваза",
    76: "ножницы",
    77: "плюшевый мишка",
    78: "фен",
    79: "зубная щетка"
}

def translate_class(class_id):
    """Переводит числовой ID класса в русское название"""
    return CLASS_TRANSLATIONS.get(int(class_id), f"Класс {class_id}")

torch.serialization.add_safe_globals([DetectionModel, torch.nn.modules.container.Sequential])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Папка для загрузок создана: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Ошибка при создании папки для загрузок: {str(e)}")

try:
    logger.info("Загрузка модели YOLO...")
    model = YOLO("yolo11x.pt", task='detect')
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {str(e)}")
    raise

def get_model_classes():
    """Получает список классов модели"""
    try:
        translated_classes = {id: translate_class(id) for id in model.names}
        return translated_classes
    except Exception as e:
        logger.error(f"Ошибка при получении классов модели: {str(e)}")
        return {}

def create_visualizations(predictions):
    try:
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        confidences = [pred['confidence'] for pred in predictions]
        sns.histplot(confidences, bins=10, color='skyblue')
        plt.title('Распределение уверенности')
        plt.xlabel('Уверенность (%)')
        plt.ylabel('Количество объектов')

        plt.subplot(232)
        class_counts = {}
        for pred in predictions:
            class_name = pred['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        sorted_classes = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
        plt.pie(sorted_classes.values(), labels=sorted_classes.keys(), autopct='%1.1f%%')
        plt.title('Распределение классов')

        plt.subplot(233)
        areas = []
        for pred in predictions:
            coords = pred['coordinates']
            area = (coords['x2'] - coords['x1']) * (coords['y2'] - coords['y1'])
            areas.append(area)
        
        sns.histplot(areas, bins=10, color='lightgreen')
        plt.title('Распределение размеров объектов')
        plt.xlabel('Площадь (пиксели)')
        plt.ylabel('Количество объектов')

        plt.subplot(234)
        aspect_ratios = []
        for pred in predictions:
            coords = pred['coordinates']
            width = coords['x2'] - coords['x1']
            height = coords['y2'] - coords['y1']
            aspect_ratio = width / height if height != 0 else 0
            aspect_ratios.append(aspect_ratio)
        
        sns.histplot(aspect_ratios, bins=10, color='salmon')
        plt.title('Распределение соотношений сторон')
        plt.xlabel('Соотношение сторон (ширина/высота)')
        plt.ylabel('Количество объектов')

        plt.subplot(235)
        quadrants = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        for pred in predictions:
            coords = pred['coordinates']
            center_x = (coords['x1'] + coords['x2']) / 2
            center_y = (coords['y1'] + coords['y2']) / 2
            if center_x > 0.5 and center_y > 0.5:
                quadrants['Q1'] += 1
            elif center_x <= 0.5 and center_y > 0.5:
                quadrants['Q2'] += 1
            elif center_x <= 0.5 and center_y <= 0.5:
                quadrants['Q3'] += 1
            else:
                quadrants['Q4'] += 1
        
        plt.bar(quadrants.keys(), quadrants.values(), color='lightblue')
        plt.title('Распределение объектов по квадрантам')
        plt.xlabel('Квадрант')
        plt.ylabel('Количество объектов')

        plt.subplot(236)
        class_sizes = {}
        for pred in predictions:
            class_name = pred['class']
            coords = pred['coordinates']
            area = (coords['x2'] - coords['x1']) * (coords['y2'] - coords['y1'])
            if class_name not in class_sizes:
                class_sizes[class_name] = []
            class_sizes[class_name].append(area)
        
        avg_sizes = {cls: sum(sizes)/len(sizes) for cls, sizes in class_sizes.items()}
        plt.bar(avg_sizes.keys(), avg_sizes.values(), color='lightgreen')
        plt.title('Средний размер объектов по классам')
        plt.xlabel('Класс')
        plt.ylabel('Средняя площадь (пиксели)')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plt.close()

        graph_url = base64.b64encode(img.getvalue()).decode()
        return f'data:image/png;base64,{graph_url}'

    except Exception as e:
        logger.error(f"Ошибка при создании графиков: {str(e)}")
        return None

@app.route('/')
def index():
    try:
        classes = get_model_classes()
        return render_template('index.html', classes=classes)
    except Exception as e:
        logger.error(f"Ошибка при загрузке главной страницы: {str(e)}")
        return "Произошла ошибка при загрузке страницы", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.warning("Файл не найден в запросе")
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Имя файла пустое")
            return jsonify({'error': 'Файл не выбран'}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logger.info(f"Сохранение файла: {filepath}")
            file.save(filepath)
            
            if not os.path.exists(filepath):
                logger.error(f"Файл не был сохранен: {filepath}")
                return jsonify({'error': 'Ошибка при сохранении файла'}), 500

            logger.info("Выполнение предсказания...")
            results = model(filepath)
            result = results[0]
            
            predictions = []
            boxes = result.boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = translate_class(class_id)
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 2),
                        'coordinates': {
                            'x1': round(x1),
                            'y1': round(y1),
                            'x2': round(x2),
                            'y2': round(y2)
                        }
                    })
                except Exception as e:
                    logger.error(f"Ошибка при обработке бокса: {str(e)}")
                    continue

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
            shutil.copy2(filepath, original_path)

            marked_path = os.path.join(app.config['UPLOAD_FOLDER'], f'marked_{filename}')
            result.save(marked_path)

            if not os.path.exists(original_path) or not os.path.exists(marked_path):
                logger.error("Ошибка при сохранении изображений")
                return jsonify({'error': 'Ошибка при сохранении изображений'}), 500

            graph_url = create_visualizations(predictions)

            return jsonify({
                'predictions': predictions,
                'original_image_url': f'/static/uploads/original_{filename}',
                'marked_image_url': f'/static/uploads/marked_{filename}',
                'graph_url': graph_url
            })

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Произошла ошибка при обработке изображения'}), 500

if __name__ == '__main__':
    try:
        logger.info("Запуск приложения...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        logger.error(traceback.format_exc()) 