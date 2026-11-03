import numpy as np

class MedianFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def apply(self, image_array):
        rows, cols = image_array.shape
        result = np.zeros_like(image_array)
        pad = self.kernel_size // 2
        padded_img = np.pad(image_array, pad, mode='edge')
        
        for i in range(rows):
            for j in range(cols):
                window = padded_img[i:i + self.kernel_size, j:j + self.kernel_size]
                result[i, j] = np.median(window)
        return result

class SobelFilter:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def apply(self, image_array):
        """
        Реалізація фільтра Собеля вручну.
        Використовує ядра згортки для знаходження градієнтів по осях X та Y.
        """
        rows, cols = image_array.shape
        result = np.zeros_like(image_array, dtype=float)
        
        Kx = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])
        
        Ky = np.array([[-1, -2, -1], 
                       [ 0,  0,  0], 
                       [ 1,  2,  1]])
        
        pad = 1  
        padded_img = np.pad(image_array, pad, mode='edge').astype(float)
        
        for i in range(rows):
            for j in range(cols):
                window = padded_img[i:i+3, j:j+3]
                
                gx = np.sum(window * Kx)
                gy = np.sum(window * Ky)
                
                magnitude = np.sqrt(gx**2 + gy**2)
                result[i, j] = magnitude
                
        result = (result / np.max(result)) * 255
        return result.astype(np.uint8)
class HistogramEqualization:
    def __init__(self):
        pass

    def apply(self, image_array):
        """
        Реалізація гістограмного вирівнювання вручну.
        """
        hist, bins = np.histogram(image_array.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
        result = cdf_final[image_array]

        return result

class OtsuSegmentation:
    def __init__(self):
        pass

    def apply(self, image_array):
        """
        Реалізація методу Оцу для автоматичного знаходження порогу бінаризації.
        Мінімізує внутрішньокласову дисперсію шляхом максимізації міжкласової.
        """
        hist, _ = np.histogram(image_array.flatten(), 256, [0, 256])
        total_pixels = image_array.size

        current_max_variance = 0
        best_threshold = 0
        sum_total = np.dot(np.arange(256), hist)
        
        sum_background = 0
        weight_background = 0

        for t in range(256):
            weight_background += hist[t]
            if weight_background == 0:
                continue
                
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * hist[t]
            
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            if variance_between > current_max_variance:
                current_max_variance = variance_between
                best_threshold = t

        binary_result = (image_array > best_threshold).astype(np.uint8) * 255
        return binary_result

class MorphologicalErosion:
    def __init__(self, kernel_size, iterations):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def apply(self, image_array):
        """
        Реалізація морфологічної ерозії (звуження об'єктів).
        Знаходить локальний мінімум у вікні ковзання.
        """
        result = image_array.copy()
        pad = self.kernel_size // 2
        
        for _ in range(self.iterations):
            padded_img = np.pad(result, pad, mode='constant', constant_values=255)
            temp_result = np.zeros_like(result)
            
            rows, cols = result.shape
            for i in range(rows):
                for j in range(cols):
                    window = padded_img[i:i + self.kernel_size, j:j + self.kernel_size]
                    temp_result[i, j] = np.min(window)
            result = temp_result
            
        return result