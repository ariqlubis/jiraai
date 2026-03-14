import unittest

from agent.attachments import parse_page_spec, _sanitize_pdf_path


class TestParsePageSpec(unittest.TestCase):
    def test_parse_page_spec_all(self):
        self.assertEqual(parse_page_spec("all", max_pages=3), [0, 1, 2])

    def test_parse_page_spec_ranges_and_list(self):
        pages = parse_page_spec("1-3,5,7-8")
        self.assertEqual(pages, [0, 1, 2, 4, 6, 7])

    def test_parse_page_spec_invalid_range(self):
        with self.assertRaises(ValueError):
            parse_page_spec("3-1")


class TestSanitizePdfPath(unittest.TestCase):
    def test_sanitize_trailing_punctuation(self):
        path, suffix = _sanitize_pdf_path("/tmp/prd.pdf.")
        self.assertEqual(path, "/tmp/prd.pdf")
        self.assertEqual(suffix, ".")

    def test_sanitize_keeps_valid_path(self):
        path, suffix = _sanitize_pdf_path("/tmp/prd.pdf")
        self.assertEqual(path, "/tmp/prd.pdf")
        self.assertEqual(suffix, "")


if __name__ == "__main__":
    unittest.main()
