import unittest
import json
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from donation import ROOT, load_data, metrics, predict_decision, rank_file


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df,_=load_data()
        cls.splits=json.loads((ROOT/'results'/'splits.json').read_text())

    def test_test_is_held_out(self):
        dev=set(self.splits['dev']);test=set(self.splits['test'])
        self.assertFalse(dev & test)
        self.assertEqual(dev | test,set(range(748)))
        seen=[]
        for part in self.splits['outer']:
            train=set(part['train']);validation=set(part['validation'])
            self.assertFalse(train & validation)
            self.assertFalse((train | validation) & test)
            self.assertEqual(train | validation,dev)
            seen.extend(validation)
        self.assertTrue(all(seen.count(i)==5 for i in dev))

    def test_group_profiles_are_separated(self):
        parts=json.loads((ROOT/'results'/'group_splits.json').read_text())
        for part in parts:
            train=set(map(tuple,self.df.iloc[part['train']][['R','F','M','T']].values))
            validation=set(map(tuple,self.df.iloc[part['validation']][['R','F','M','T']].values))
            self.assertFalse(train & validation)

    def test_persisted_models_and_results_agree(self):
        pred=pd.read_csv(ROOT/'results'/'test_predictions.csv')
        scores=pd.read_csv(ROOT/'results'/'test_scores.csv').set_index('model')
        for name in scores.index:
            model=joblib.load(ROOT/'models'/f'{name}.joblib')
            p=model.predict_proba(self.df.iloc[pred.id][['R','F','M','T']])[:,1]
            np.testing.assert_allclose(p,pred[name],atol=1e-12)
            for key,value in metrics(pred.y,p,pred.id).items():
                self.assertAlmostEqual(value,scores.loc[name,key],places=12)

    def test_rbf_formula_matches_library(self):
        calibrated=joblib.load(ROOT/'models'/'svm.joblib').calibrated_classifiers_[0]
        X=self.df.iloc[self.splits['test']][['R','F','M','T']]
        np.testing.assert_allclose(predict_decision(calibrated.estimator,X),
            calibrated.estimator.decision_function(X),atol=1e-8)

    def test_exact_budget_even_with_ties_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/'input.csv';dst=Path(folder)/'output.csv'
            pd.DataFrame({'R':[2,2,np.nan],'F':[3,3,1],'T':[20,20,10]}).to_csv(src,index=False)
            rank_file(src,1,dst)
            result=pd.read_csv(dst)
            self.assertEqual(int(result.invite.sum()),1)
            self.assertTrue(result.probability.between(0,1).all())
            self.assertTrue(result.probability.is_monotonic_decreasing)
            with self.assertRaises(ValueError):rank_file(src,4,dst)


if __name__=='__main__':unittest.main()
