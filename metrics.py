import numpy as np

def calculate_mse(original, processed):
    """Обчислення MSE (Середньоквадратичної помилки)"""
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    mse = np.mean((orig - proc) ** 2)
    return mse

def calculate_psnr(mse, max_pixel=255.0):
    """Обчислення PSNR (Пікового відношення сигнал/шум)"""
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(original, processed):
    """Обчислення глобального SSIM (Індексу структурної подібності)"""
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = np.mean(orig)
    mu_y = np.mean(proc)

    sigma_x2 = np.var(orig)
    sigma_y2 = np.var(proc)
    
    sigma_xy = np.mean((orig - mu_x) * (proc - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    
    return num / den