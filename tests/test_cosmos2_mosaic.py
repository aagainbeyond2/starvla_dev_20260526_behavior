import unittest

from PIL import Image

from starVLA.model.modules.world_model.CosmoPredict2 import _CosmoPredict2_Interface


def _solid(size, color):
    return Image.new("RGB", size, color)


class Cosmos2MosaicTests(unittest.TestCase):
    def test_layout_and_view_order(self):
        head = _solid((224, 224), (255, 0, 0))
        left_wrist = _solid((224, 224), (0, 255, 0))
        right_wrist = _solid((224, 224), (0, 0, 255))

        mosaic = _CosmoPredict2_Interface._compose_cosmos3_mosaic(
            [head, left_wrist, right_wrist]
        )

        self.assertEqual(mosaic.mode, "RGB")
        self.assertEqual(mosaic.size, (224, 336))
        self.assertEqual(mosaic.getpixel((111, 111)), (255, 0, 0))
        self.assertEqual(mosaic.getpixel((55, 279)), (0, 255, 0))
        self.assertEqual(mosaic.getpixel((167, 279)), (0, 0, 255))

    def test_requires_exactly_three_views(self):
        for num_views in (0, 1, 2, 4):
            views = [
                _solid((224, 224), (index, index, index))
                for index in range(num_views)
            ]
            with self.subTest(num_views=num_views):
                with self.assertRaisesRegex(ValueError, "requires exactly 3 views"):
                    _CosmoPredict2_Interface._compose_cosmos3_mosaic(views)


if __name__ == "__main__":
    unittest.main()
