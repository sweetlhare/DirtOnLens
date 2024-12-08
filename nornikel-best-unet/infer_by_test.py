import os
import sys
import json
import base64
import cv2
import numpy as np
import onnxruntime

dataset_path, output_path = sys.argv[1:]

def initialize_onnx_model(model_path):
    """Initialize ONNX Runtime inference session"""
    try:
        # Опционально используем CUDA если доступна
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() \
            else ['CPUExecutionProvider']
            
        session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )
        print(f'Используемый провайдер: {session.get_providers()}')
        return session
    except Exception as e:
        print(f"Ошибка инициализации ONNX модели: {str(e)}")
        raise

def infer_image(session, image_path):
    """Inference single image using ONNX Runtime"""
    try:
        # Чтение и изменение размера изображения
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))

        # Предобработка как в PyTorch версии
        img = image.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Получаем имя входного тензора
        input_name = session.get_inputs()[0].name
        
        # ONNX Runtime inference
        outputs = session.run(None, {input_name: img})[0]
        
        # Постобработка
        preds_sig = 1 / (1 + np.exp(-outputs))  # sigmoid
        mask = (preds_sig > 0.5).astype(np.float32)
        
        return mask[0, 0]  # Возвращаем первое изображение, первый канал

    except Exception as e:
        print(f"Ошибка при инференсе изображения: {str(e)}")
        raise

def process_dataset(session, dataset_path, output_path, visualization_dir="./mask_visualization"):
    """Process entire dataset and save results"""
    os.makedirs(visualization_dir, exist_ok=True)
    results_dict = {}

    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(".jpg"):
            try:
                image_path = os.path.join(dataset_path, image_name)
                original_image = cv2.imread(image_path)
                original_size = original_image.shape[:2]

                # Получаем маску
                mask = infer_image(session, image_path)

                # Постобработка маски
                mask_uint8 = ((mask > 0.5) * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (original_size[1], original_size[0]))
                mask_resized = ((mask_resized > 127) * 255).astype(np.uint8)

                # Кодируем маску в base64
                success, encoded_img = cv2.imencode(".png", mask_resized)
                if not success:
                    print(f"Ошибка при кодировании маски для {image_name}")
                    continue

                encoded_str = base64.b64encode(encoded_img).decode('utf-8')
                results_dict[image_name] = encoded_str

                # Опционально: визуализация
                visualize_mask(original_image, mask_resized, image_name)

            except Exception as e:
                print(f"Ошибка при обработке {image_name}: {str(e)}")

    return results_dict

def visualize_mask(original_image, mask, image_name):
    """Visualize mask overlay on original image"""
    visualization = original_image.copy()
    visualization[mask > 0] = [0, 0, 255]  # Red overlay for mask

    overlay = original_image.copy()
    alpha = 0.5
    cv2.addWeighted(visualization, alpha, original_image, 1 - alpha, 0, overlay)
    
    save_path = os.path.join("./mask_visualization", f"vis_{image_name}")
    mask_save_path = os.path.join("./mask_visualization", f"mask_{image_name}")
    
    cv2.imwrite(save_path, overlay)
    cv2.imwrite(mask_save_path, mask)

def main():
    try:
        # Инициализация модели
        session = initialize_onnx_model("segmentation_model.onnx")
        
        # Обработка датасета
        results = process_dataset(session, dataset_path, output_path)
        
        # Сохранение результатов
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"Результаты успешно сохранены в {output_path}")
            
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()