import json
import os
import csv
import numpy as np
from PIL import Image
from processors import (
    MedianFilter, SobelFilter, HistogramEqualization, 
    OtsuSegmentation, MorphologicalErosion
)
from metrics import calculate_mse, calculate_psnr, calculate_ssim

def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        return json.load(file)

def save_metrics(writer, filename, method_name, original, processed):
    mse = calculate_mse(original, processed)
    psnr = calculate_psnr(mse)
    ssim = calculate_ssim(original, processed)
    writer.writerow([filename, method_name, round(mse, 4), round(psnr, 4), round(ssim, 4)])

def main():
    config = load_config()
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    metrics_file = os.path.join(output_dir, config.get("metrics_file", "metrics.csv"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods_config = config["methods"]
    median = MedianFilter(methods_config["median_filter"]["kernel_size"])
    sobel = SobelFilter(methods_config["sobel_filter"]["kernel_size"])
    hist_eq = HistogramEqualization()
    otsu = OtsuSegmentation()
    erosion = MorphologicalErosion(
        methods_config["erosion"]["kernel_size"], 
        methods_config["erosion"]["iterations"]
    )
    
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Method", "MSE", "PSNR", "SSIM"])
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                filepath = os.path.join(input_dir, filename)
                
                try:
                    img = Image.open(filepath).convert('L')
                    img_array = np.array(img)
                    
                    if img_array.size == 0 or len(img_array.shape) < 2:
                        print(f"Пропуск {filename}: файл порожній або не є коректним зображенням.")
                        continue
                        
                    if img_array.shape[0] < 3 or img_array.shape[1] < 3:
                        print(f"Пропуск {filename}: зображення занадто мале для обробки ядром 3x3.")
                        continue
                        
                    print(f"Обробка {filename}...")
                    
                    filtered_array = median.apply(img_array)
                    Image.fromarray(filtered_array).save(os.path.join(output_dir, f"median_{filename}"))
                    save_metrics(writer, filename, "Median Filter", img_array, filtered_array)
                    
                    sobel_array = sobel.apply(img_array)
                    Image.fromarray(sobel_array).save(os.path.join(output_dir, f"sobel_{filename}"))
                    save_metrics(writer, filename, "Sobel Filter", img_array, sobel_array)
                    
                    hist_array = hist_eq.apply(img_array)
                    Image.fromarray(hist_array).save(os.path.join(output_dir, f"hist_{filename}"))
                    save_metrics(writer, filename, "Histogram Equalization", img_array, hist_array)
                    
                    otsu_array = otsu.apply(img_array)
                    Image.fromarray(otsu_array).save(os.path.join(output_dir, f"otsu_{filename}"))
                    save_metrics(writer, filename, "Otsu Segmentation", img_array, otsu_array)
                    
                    erosion_array = erosion.apply(img_array)
                    Image.fromarray(erosion_array).save(os.path.join(output_dir, f"erosion_{filename}"))
                    save_metrics(writer, filename, "Erosion", img_array, erosion_array)
                    
                except Exception as e:
                    print(f"Помилка при обробці файлу {filename}: {e}")

if __name__ == "__main__":
    main()