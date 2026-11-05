import json, unittest
from run_eval import passes, mcnemar_exact

class T(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(passes("Overdispersion matters", ["overdispersion"]))
        self.assertFalse(passes("no idea", ["overdispersion"]))
    def test_mcnemar(self):
        self.assertAlmostEqual(mcnemar_exact(5, 1), 0.21875)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
    def test_data_is_valid_jsonl(self):
        rows = [json.loads(l) for l in open("held_out.jsonl", encoding="utf-8")]
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["question"] for r in rows}), 30)
        self.assertTrue(all(r["must_include"] for r in rows))

class TJudge(unittest.TestCase):
    def test_sign_test_value(self):
        self.assertAlmostEqual(mcnemar_exact(0, 10), 0.001953125)

class TErrors(unittest.TestCase):
    def test_failed_call_never_passes(self):
        self.assertFalse(passes("ERROR: HTTP 500", ["h"]))
    def test_hard_set_has_references(self):
        rows = [json.loads(l) for l in open("held_out_hard.jsonl", encoding="utf-8")]
        self.assertEqual(len(rows), 19)
        self.assertTrue(all(r["reference"] for r in rows))

if __name__ == "__main__":
    unittest.main()
