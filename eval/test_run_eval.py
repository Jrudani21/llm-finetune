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

if __name__ == "__main__":
    unittest.main()
