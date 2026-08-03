import unittest
import re

class TestTextNormalization(unittest.TestCase):
    def test_whitespace_normalization(self):
        text = "et saa\nTätä\ttunnet katoomaan\nEt saa, et saa-a-a"
        clean = re.sub(r'[\r\n\t]+', ' ', text).strip()
        self.assertNotIn("\n", clean)
        self.assertNotIn("\t", clean)
        self.assertEqual(clean, "et saa Tätä tunnet katoomaan Et saa, et saa-a-a")

if __name__ == "__main__":
    unittest.main()
