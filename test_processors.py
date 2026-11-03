import unittest
import numpy as np
from processors import (
    MedianFilter, SobelFilter, HistogramEqualization, 
    OtsuSegmentation, MorphologicalErosion
)

class TestImageProcessors(unittest.TestCase):
    
    def test_median_filter(self):
        # Матриця з різким викидом (100) по центру
        img = np.array([[10, 10, 10],
                        [10, 100, 10],
                        [10, 10, 10]], dtype=np.uint8)
        processor = MedianFilter(kernel_size=3)
        result = processor.apply(img)
        # Медіана для вікна 3x3 з дев'яти значень (вісім 10 і одна 100) дорівнює 10
        self.assertEqual(result[1, 1], 10)

    def test_otsu_segmentation(self):
        # Матриця з чітким фоном (0) та об'єктом (200)
        img = np.array([[0, 0, 0],
                        [200, 200, 200],
                        [200, 200, 200]], dtype=np.uint8)
        processor = OtsuSegmentation()
        result = processor.apply(img)
        # Після бінаризації 200 має стати 255, а 0 залишитися 0
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[2, 2], 255)

    def test_morphological_erosion(self):
        # Матриця білого кольору (255) з чорним пікселем (0) по центру
        img = np.array([[255, 255, 255],
                        [255, 0, 255],
                        [255, 255, 255]], dtype=np.uint8)
        processor = MorphologicalErosion(kernel_size=3, iterations=1)
        result = processor.apply(img)
        # Ерозія шукає мінімум. Чорний піксель має поширитися на сусідів
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[1, 2], 0)

    def test_sobel_filter(self):
        # Матриця з вертикальною межею
        img = np.array([[0, 0, 255],
                        [0, 0, 255],
                        [0, 0, 255]], dtype=np.uint8)
        processor = SobelFilter(kernel_size=3)
        result = processor.apply(img)
        # На межі переходу (центральний стовпець) значення градієнта має бути високим
        self.assertGreater(result[1, 1], 0)

    def test_histogram_equalization(self):
        # Рівномірно тьмяна матриця
        img = np.array([[50, 50, 51],
                        [50, 52, 50],
                        [51, 50, 50]], dtype=np.uint8)
        processor = HistogramEqualization()
        result = processor.apply(img)
        # Після вирівнювання значення розтягуються на весь діапазон 0-255
        self.assertEqual(np.max(result), 255)
        self.assertEqual(np.min(result), 0)

if __name__ == '__main__':
    unittest.main()