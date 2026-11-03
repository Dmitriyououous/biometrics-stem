"""Головний модуль для запуску конвеєра обробки біометричних зображень."""
import json
import os
import csv
import numpy as np
from PIL import Image
from processors import (
    MedianFilter, SobelFilter, HistogramEqualization,
    OtsuSegmentation, MorphologicalErosion, LocalBinaryPattern
)
from metrics import calculate_mse, calculate_psnr, calculate_ssim

def load_config(config_path="config.json"):
    """Завантаження налаштувань конфігурації з JSON файлу."""
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_metrics(writer, filename, method_name, original, processed):
    """Обчислення та збереження метрик якості у файл CSV."""
    mse = calculate_mse(original, processed)
    psnr = calculate_psnr(mse)
    ssim = calculate_ssim(original, processed)
    writer.writerow([filename, method_name, round(mse, 4), round(psnr, 4), round(ssim, 4)])

def process_single_image(img_array, filename, output_dir, writer, methods):
    """Послідовне застосування всіх методів до одного зображення."""
    method_configs = [
        ("Median Filter", "median_", methods[0]),
        ("Sobel Filter", "sobel_", methods[1]),
        ("Histogram Eq", "hist_", methods[2]),
        ("Otsu Seg", "otsu_", methods[3]),
        ("Erosion", "erosion_", methods[4]),
        ("LBP Texture", "lbp_", methods[5])
    ]

    for method_name, prefix, processor in method_configs:
        res_array = processor.apply(img_array)
        Image.fromarray(res_array).save(os.path.join(output_dir, f"{prefix}{filename}"))
        save_metrics(writer, filename, method_name, img_array, res_array)

def main():
    """Основна функція виконання програми."""
    config = load_config()
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    metrics_file = os.path.join(output_dir, config.get("metrics_file", "metrics.csv"))

    os.makedirs(output_dir, exist_ok=True)
    methods_cfg = config["methods"]
    
    methods = (
        MedianFilter(methods_cfg["median_filter"]["kernel_size"]),
        SobelFilter(methods_cfg["sobel_filter"]["kernel_size"]),
        HistogramEqualization(),
        OtsuSegmentation(),
        MorphologicalErosion(
            methods_cfg["erosion"]["kernel_size"],
            methods_cfg["erosion"]["iterations"]
        ),
        LocalBinaryPattern()
    )

    with open(metrics_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Method", "MSE", "PSNR", "SSIM"])

        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                continue

            filepath = os.path.join(input_dir, filename)

            try:
                img = Image.open(filepath).convert('L')
                img_array = np.array(img)

                if img_array.size == 0 or len(img_array.shape) < 2:
                    print(f"Пропуск {filename}: некоректне зображення.")
                    continue

                if img_array.shape[0] < 3 or img_array.shape[1] < 3:
                    print(f"Пропуск {filename}: зображення занадто мале.")
                    continue

                print(f"Обробка {filename}...")
                process_single_image(img_array, filename, output_dir, writer, methods)

            except Exception as e: 
                print(f"Помилка при обробці файлу {filename}: {e}")

if __name__ == "__main__":
    main()