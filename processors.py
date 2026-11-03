"""Модуль з класами алгоритмів для обробки біометричних зображень."""
# pylint: disable=too-few-public-methods

import numpy as np

class MedianFilter:
    """Клас для застосування просторової медіанної фільтрації."""
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def apply(self, image_array):
        """Виконання медіанного фільтра."""
        pad = self.kernel_size // 2
        padded_img = np.pad(image_array, pad, mode='edge')
        result = np.zeros_like(image_array)
        rows, cols = image_array.shape

        for i in range(rows):
            for j in range(cols):
                window = padded_img[i:i + self.kernel_size, j:j + self.kernel_size]
                result[i, j] = np.median(window)

        return result

class SobelFilter:
    """Клас для виявлення контурів за допомогою оператора Собеля."""
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def apply(self, image_array):
        """Виконання фільтра Собеля."""
        rows, cols = image_array.shape
        result = np.zeros_like(image_array, dtype=float)

        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        kernel_y = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]])

        pad = 1
        padded_img = np.pad(image_array, pad, mode='edge').astype(float)

        for i in range(rows):
            for j in range(cols):
                window = padded_img[i:i+3, j:j+3]
                gx = np.sum(window * kernel_x)
                gy = np.sum(window * kernel_y)
                result[i, j] = np.sqrt(gx**2 + gy**2)

        result = (result / np.max(result)) * 255
        return result.astype(np.uint8)

class HistogramEqualization:
    """Клас для виконання гістограмного вирівнювання контрастності."""
    def __init__(self):
        pass

    def apply(self, image_array):
        """Виконання гістограмного вирівнювання."""
        hist, _ = np.histogram(image_array.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        
        num = (cdf_masked - cdf_masked.min()) * 255
        den = cdf_masked.max() - cdf_masked.min()
        cdf_masked = num / den
        
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
        return cdf_final[image_array]

class OtsuSegmentation:
    """Клас для бінаризації зображення за методом Оцу."""
    def __init__(self):
        pass

    def apply(self, image_array):
        """Виконання сегментації методу Оцу."""
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
            mean_bg = sum_background / weight_background
            mean_fg = (sum_total - sum_background) / weight_foreground

            variance_between = weight_background * weight_foreground * (mean_bg - mean_fg) ** 2

            if variance_between > current_max_variance:
                current_max_variance = variance_between
                best_threshold = t

        return (image_array > best_threshold).astype(np.uint8) * 255

class MorphologicalErosion:
    """Клас для виконання морфологічної ерозії (звуження)."""
    def __init__(self, kernel_size, iterations):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def apply(self, image_array):
        """Виконання морфологічної ерозії."""
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

class LocalBinaryPattern:
    """Клас для виділення текстурних ознак за алгоритмом LBP."""
    def __init__(self):
        pass

    def apply(self, image_array):
        """Виконання перетворення Local Binary Pattern (векторизовано)."""
        padded = np.pad(image_array, 1, mode='edge')
        center = padded[1:-1, 1:-1]
        out = np.zeros_like(center, dtype=np.uint8)

        out += (padded[0:-2, 0:-2] >= center).astype(np.uint8) * 128
        out += (padded[0:-2, 1:-1] >= center).astype(np.uint8) * 64
        out += (padded[0:-2, 2:] >= center).astype(np.uint8) * 32
        out += (padded[1:-1, 2:] >= center).astype(np.uint8) * 16
        out += (padded[2:, 2:] >= center).astype(np.uint8) * 8
        out += (padded[2:, 1:-1] >= center).astype(np.uint8) * 4
        out += (padded[2:, 0:-2] >= center).astype(np.uint8) * 2
        out += (padded[1:-1, 0:-2] >= center).astype(np.uint8) * 1

        return out