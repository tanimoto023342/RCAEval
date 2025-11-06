import unittest

class TestPillowImport(unittest.TestCase):
    def test_import_pillow(self):
        try:
            from PIL import Image
            self.assertTrue(True)
        except ImportError:
            self.fail("Pillow is not installed")

if __name__ == '__main__':
    unittest.main()