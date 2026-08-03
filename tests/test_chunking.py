import unittest
from app.model_manager import split_text_into_chunks

class TestSplitTextIntoChunks(unittest.TestCase):
    def test_short_text(self):
        text = "Hello world!"
        chunks = split_text_into_chunks(text, max_chars=100)
        self.assertEqual(chunks, ["Hello world!"])

    def test_word_packing_up_to_100_chars(self):
        text = "word " * 25  # 5 * 25 = 125 chars total
        chunks = split_text_into_chunks(text, max_chars=100)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_long_single_string_splitting(self):
        # A single continuous string of 150 characters without spaces
        long_string = "a" * 150
        chunks = split_text_into_chunks(long_string, max_chars=100)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 100)
        self.assertEqual(len(chunks[1]), 50)

    def test_newlines_and_tabs_normalized(self):
        text = "et saa\nTätä\ttunnet katoomaan\nEt saa, et saa-a-a"
        chunks = split_text_into_chunks(text, max_chars=100)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertNotIn("\n", chunk)
            self.assertNotIn("\t", chunk)
            self.assertLessEqual(len(chunk), 100)

    def test_empty_string(self):
        self.assertEqual(split_text_into_chunks(""), [])
        self.assertEqual(split_text_into_chunks("   "), [])

    def test_multiple_paragraphs(self):
        text = (
            "OmniVoice is a zero-shot text-to-speech system. It provides fast and expressive speech synthesis! "
            "We want to chunk long input texts into small segments of whole words. "
            "Each segment must be at most one hundred characters long, so that generation is reliable and fast. "
            "Finally, all generated audio segments are concatenated together into a single audio response."
        )
        chunks = split_text_into_chunks(text, max_chars=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

if __name__ == "__main__":
    unittest.main()
