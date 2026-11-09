"""Модуль для візуалізації метрик якості обробки зображень."""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics():
    """Зчитує metrics.csv та будує графіки порівняння метрик."""
    csv_path = "images/output/metrics.csv"
    
    if not os.path.exists(csv_path):
        print("Файл metrics.csv не знайдено! Спочатку запустіть main.py.")
        return

    methods_data = {}
    
    # Зчитуємо дані з CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row['Method']
            if m not in methods_data:
                methods_data[m] = {'PSNR': [], 'SSIM': []}
            methods_data[m]['PSNR'].append(float(row['PSNR']))
            methods_data[m]['SSIM'].append(float(row['SSIM']))

    # Обчислюємо середні значення для кожного методу
    names = list(methods_data.keys())
    avg_psnr = [np.mean(methods_data[m]['PSNR']) for m in names]
    avg_ssim = [np.mean(methods_data[m]['SSIM']) for m in names]

    # графік для PSNR
    plt.figure(figsize=(10, 5))
    plt.bar(names, avg_psnr, color='cornflowerblue', edgecolor='black')
    plt.title('Середнє значення PSNR для кожного методу (Більше = Краще)')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("images/output/chart_psnr.png") # Зберігаємо як картинку
    plt.show()

    #графік для SSIM
    plt.figure(figsize=(10, 5))
    plt.bar(names, avg_ssim, color='lightgreen', edgecolor='black')
    plt.title('Середнє значення SSIM (Ближче до 1.0 = Краще)')
    plt.ylabel('SSIM')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("images/output/chart_ssim.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics()