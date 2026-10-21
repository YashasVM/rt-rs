#!/usr/bin/env python3
"""Model-free report validation tests. These do not simulate genuine inference."""
import copy
import unittest

from validate import validate


class ValidateReportTest(unittest.TestCase):
    def setUp(self):
        self.report = {
            "status": "OK",
            "kind": "instrumented_native_diagnostic_not_benchmark",
            "factory_calls": 1, "sample_calls": 1,
            "resolved_threads": 2, "resolved_max_output_tokens": 16,
            "resolved_engine_settings": "synthetic fixture",
            "resolved_conversation_session_config": "synthetic fixture",
            "resolved_prompt_template": "synthetic fixture",
            "response": {"content": "synthetic fixture"},
            "factories": [{"status": "OK", "backend": 3, "batch_size": 1,
                           "sampling": dict(type=2, k=1, p=1, temperature=1,
                                            seed=0, has_seed=True)}],
            "samples": [{"status": "OK", "dtype": "float32", "shape": [1, 1, 10],
                         "token_ids": [3], "sampling": dict(type=2, k=1, p=1, temperature=1,
                                                           seed=0, has_seed=True)}],
        }

    def test_accepts_supported_dtypes(self):
        for dtype in ("float32", "float16"):
            self.report["samples"][0]["dtype"] = dtype
            result = validate(self.report)
            self.assertEqual(result["sampled_token_ids"], [3])
            self.assertEqual(result["observed_dtypes"], [dtype])

    def test_rejects_invalid_reports(self):
        mutations = [
            lambda r: r.update(sample_calls=0),
            lambda r: r.update(factory_calls=0),
            lambda r: r.update(status="INTERNAL: synthetic fixture"),
            lambda r: r.update(resolved_threads=4),
            lambda r: r.update(resolved_max_output_tokens=32),
            lambda r: r.update(resolved_prompt_template=""),
            lambda r: r["samples"][0].update(dtype="other"),
            lambda r: r["samples"][0].update(token_ids=[99]),
            lambda r: r["samples"][0].update(shape=[1, 2, 10]),
            lambda r: r["samples"][0]["sampling"].update(k=2),
            lambda r: r["factories"][0].update(backend=4),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                report = copy.deepcopy(self.report)
                mutate(report)
                with self.assertRaises(ValueError):
                    validate(report)


if __name__ == "__main__":
    unittest.main()
