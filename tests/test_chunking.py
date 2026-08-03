import unittest
from app.model_manager import split_text_into_chunks

class TestSplitTextIntoChunks(unittest.TestCase):
    def test_short_text(self):
        text = "Hello world!"
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertEqual(chunks, ["Hello world!"])

    def test_two_short_sentences(self):
        text = "Hello world! This is a test."
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world! This is a test.")

    def test_three_sentences_splits_by_sentence_count(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "First sentence here. Second sentence here.")
        self.assertEqual(chunks[1], "Third sentence here.")

    def test_max_character_limit_per_chunk(self):
        s1 = "This is a sentence that is around forty-five characters long."
        s2 = "Here is another sentence that is also around forty-five characters long."
        text = f"{s1} {s2}"
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_long_single_sentence_clause_splitting(self):
        text = "This is a single very long sentence, which contains multiple clauses separated by commas, making it exceed the hundred character limit easily."
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_empty_string(self):
        self.assertEqual(split_text_into_chunks(""), [])
        self.assertEqual(split_text_into_chunks("   "), [])

    def test_newlines_handling(self):
        text = "et saa\nTätä tunnet katoomaan\nEt saa, et saa-a-a"
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertNotIn("\n", chunk)
            self.assertLessEqual(len(chunk), 100)

    def test_long_text_paragraph(self):
        text = (
            "OmniVoice is a zero-shot text-to-speech system. It provides fast and expressive speech synthesis! "
            "We want to chunk long input texts into small segments of one to two sentences. "
            "Each segment must be at most one hundred characters long, so that generation is reliable and fast. "
            "Finally, all generated audio segments are concatenated together into a single audio response."
        )
        chunks = split_text_into_chunks(text, max_chars=100, max_sentences=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

if __name__ == "__main__":
    unittest.main()
