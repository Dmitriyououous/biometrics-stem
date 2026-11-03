"""Модуль для виконання автотестування алгоритмів обробки зображень."""
import unittest
import numpy as np
from processors import (
    MedianFilter, SobelFilter, HistogramEqualization,
    OtsuSegmentation, MorphologicalErosion, LocalBinaryPattern
)

class TestImageProcessors(unittest.TestCase):
    """Клас автотестів для перевірки базової логіки процесорів."""

    def test_median_filter(self):
        """Перевірка ізоляції екстремальних пікселів медіанним фільтром."""
        img = np.array([[10, 10, 10],
                        [10, 100, 10],
                        [10, 10, 10]], dtype=np.uint8)
        processor = MedianFilter(kernel_size=3)
        result = processor.apply(img)
        self.assertEqual(result[1, 1], 10)

    def test_otsu_segmentation(self):
        """Перевірка правильного визначення порогу бінаризації методом Оцу."""
        img = np.array([[0, 0, 0],
                        [200, 200, 200],
                        [200, 200, 200]], dtype=np.uint8)
        processor = OtsuSegmentation()
        result = processor.apply(img)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[2, 2], 255)

    def test_morphological_erosion(self):
        """Перевірка звуження об'єкта під час морфологічної ерозії."""
        img = np.array([[255, 255, 255],
                        [255, 0, 255],
                        [255, 255, 255]], dtype=np.uint8)
        processor = MorphologicalErosion(kernel_size=3, iterations=1)
        result = processor.apply(img)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[1, 2], 0)

    def test_sobel_filter(self):
        """Перевірка виявлення вертикальної межі фільтром Собеля."""
        img = np.array([[0, 0, 255],
                        [0, 0, 255],
                        [0, 0, 255]], dtype=np.uint8)
        processor = SobelFilter(kernel_size=3)
        result = processor.apply(img)
        self.assertGreater(result[1, 1], 0)

    def test_histogram_equalization(self):
        """Перевірка розширення діапазону контрастності гістограмним вирівнюванням."""
        img = np.array([[50, 50, 51],
                        [50, 52, 50],
                        [51, 50, 50]], dtype=np.uint8)
        processor = HistogramEqualization()
        result = processor.apply(img)
        self.assertEqual(np.max(result), 255)
        self.assertEqual(np.min(result), 0)
    def test_local_binary_pattern(self):
        """Перевірка обчислення текстури LBP для простої матриці."""
        img = np.array([[0,   0,   0],
                        [0, 100,   0],
                        [0,   0,   0]], dtype=np.uint8)
        processor = LocalBinaryPattern()
        result = processor.apply(img)
        self.assertEqual(result[1, 1], 0)

if __name__ == '__main__':
    unittest.main()