import unittest, subprocess, sys, json
from pathlib import Path
BASE=Path(__file__).resolve().parent
class ProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable,str(BASE/"main.py")],cwd=BASE,check=True,capture_output=True,text=True)
        cls.metrics=json.loads((BASE/"results/metrics.json").read_text())
    def test_split_sizes(self):
        m=self.metrics
        self.assertEqual(m["train_rows"]+m["test_rows"],m["rows"])
        self.assertEqual(m["seed"],42)
    def test_metrics_and_baseline(self):
        rows=self.metrics["held_out"]
        self.assertEqual(rows[0]["model"],"baseline")
        self.assertEqual(len(rows),2)
        for row in rows:
            for key,value in row.items():
                if key!="model":self.assertTrue(__import__("math").isfinite(value))
    def test_predictions_and_figures(self):
        import pandas as pd
        predictions=pd.read_csv(BASE/"results/test_predictions.csv")
        self.assertEqual(len(predictions),self.metrics["test_rows"])
        self.assertFalse(predictions.isna().any().any())
        self.assertEqual(predictions.source_row.nunique(),len(predictions))
        self.assertTrue(any((BASE/"results").glob("*.png")))
if __name__=="__main__":unittest.main()
