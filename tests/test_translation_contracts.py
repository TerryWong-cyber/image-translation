import importlib.util
import unittest


PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None

if PYDANTIC_AVAILABLE:
    from pydantic import ValidationError

    from image_translation.contracts import TranslationCommand, TranslationItem, TranslationResult


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic is not installed")
class TranslationContractsTest(unittest.TestCase):
    def test_command_accepts_nested_object_key_and_supported_language(self):
        command = TranslationCommand(
            bucket="source-images",
            image_key="manuals/page-1.png",
            language="zh_en",
        )

        self.assertEqual(command.image_key, "manuals/page-1.png")
        self.assertEqual(command.save_bucket, None)

    def test_command_rejects_path_traversal(self):
        with self.assertRaises(ValidationError):
            TranslationCommand(bucket="source", image_key="../secret.png")

    def test_rest_payload_preserves_legacy_shape(self):
        result = TranslationResult(
            request_id="request-1",
            source_url="source/input.png",
            translated_url="target/en_zh_translated_input.png",
            data=[
                TranslationItem(
                    coordinate=[[0, 0], [10, 0], [10, 10], [0, 10]],
                    source_text="Hello",
                    translated_text="你好",
                )
            ],
            duration_ms=10,
        )

        payload = result.to_rest_payload()
        self.assertEqual(
            set(payload),
            {"status", "source_url", "translated_url", "data"},
        )
        self.assertNotIn("request_id", payload)


if __name__ == "__main__":
    unittest.main()
