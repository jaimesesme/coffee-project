import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import jev_evaluation as j

class Response:
    request_id='mock-request'
    def __init__(self,bad=False):self.bad=bad
    def model_dump(self):
        ans={q:{'type':'choice','choice':'unclear','confidence':.9,
                'probabilities':{'unclear':.9,'positive':.05,'negative':.05}} for q in j.QUALIFIERS}
        if self.bad:ans.pop(j.QUALIFIERS[0])
        return {'model':'jev-1.14.0','answers':ans,'usage':{'input_tokens':1000,'output_tokens':100}}

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.p=Path(self.t.name)/'pilot.jsonl'
        self.df=pd.DataFrame([{'review_id':str(i),'business_id':'b','text':'review '+str(i),**{q:0 for q in j.QUALIFIERS}} for i in range(100)])
        self.spec={q:{'type':'choice','instructions':'question','criteria':{k:k for k in j.CHOICE_LABELS}} for q in j.QUALIFIERS}

    def test_mapping_and_bad_output(self):
        labels,_=j.parse_answers(Response());self.assertEqual(set(labels.values()),{0})
        with self.assertRaises(ValueError):j.parse_answers(Response(True))

    def test_full_dataset_resume_and_source_change(self):
        source = self.df.head(3)[['review_id', 'business_id', 'text']].copy()
        calls = []
        def fake(**kw):
            calls.append(kw)
            return Response()
        client = SimpleNamespace(system_one=fake)
        args = (source, self.spec, client, 'jev-1.14.0', 'full_v1', self.p)
        j.run_jev(*args, stage='full', limit=1, min_interval=0)
        saved = j.run_jev(*args, stage='full', limit=None, min_interval=0)
        self.assertEqual(len(saved), 3)
        self.assertEqual(len(calls), 3)
        j.run_jev(*args, stage='full', limit=None, min_interval=0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(set(c) == {'state', 'questions'} for c in calls))
        source.loc[0, 'text'] = 'changed'
        with self.assertRaises(ValueError):
            j.run_jev(*args, stage='full', limit=None, min_interval=0)
        self.assertEqual(len(calls), 3)

    def test_full_source_validation_before_requests(self):
        client = SimpleNamespace(system_one=lambda **kw: self.fail('Unexpected API call'))
        source = self.df.head(3).copy()
        source.loc[0, 'text'] = ' '
        with self.assertRaises(ValueError):
            j.run_jev(source, self.spec, client, 'jev-1.14.0', 'full_v1', self.p,
                      stage='full', limit=None, min_interval=0)

    def test_run_resume_cost_and_no_human_leak(self):
        calls=[]
        def fake(**kw):
            calls.append(kw)
            self.assertIsInstance(kw['state'],str)
            self.assertEqual(set(kw),{'state','questions'})
            return Response()
        client=SimpleNamespace(system_one=fake)
        j.run_jev(self.df,self.spec,client,'jev-1.14.0','v1',self.p,limit=2,min_interval=0)
        j.run_jev(self.df,self.spec,client,'jev-1.14.0','v1',self.p,limit=2,min_interval=0)
        self.assertEqual(len(calls),4)
        self.assertEqual(len(j.predictions(self.p)),4)
        cost=j.timing_cost(self.p)
        self.assertAlmostEqual(cost['estimated_usd_known_usage'],4000*.042/1e6)
        self.assertGreater(cost['wall_seconds_recorded'],0)
        with self.assertRaises(ValueError):j.run_jev(self.df,self.spec,client,'jev-1.14.0','v2',self.p,limit=1,min_interval=0)
        self.assertEqual(len(calls),4)

    def test_failure_separate_and_partial_evaluation_blocked(self):
        j.run_jev(self.df,self.spec,SimpleNamespace(system_one=lambda **kw:Response(True)),
                  'jev-1.14.0','v1',self.p,limit=1,min_interval=0)
        self.assertEqual(j.predictions(self.p),[])
        self.assertEqual(len(j.read_jsonl(self.p.with_suffix('.failures.jsonl'))),1)
        self.assertEqual(j.timing_cost(self.p)['input_tokens_known'],1000)
        with self.assertRaises(ValueError):j.evaluate(self.df,[])

    def test_metrics_known_confusion(self):
        human=self.df.head(3).copy()
        human.loc[:,j.QUALIFIERS[0]]=[-1,0,1]
        rows=[{'review_id':str(i),'labels':{q:0 for q in j.QUALIFIERS}} for i in range(3)]
        report=j.evaluate(human,rows)
        first=report['summary'].iloc[0]
        self.assertEqual(first.disagreements,2)
        self.assertAlmostEqual(first.agreement_pct,100/3)
        self.assertAlmostEqual(first.macro_f1_present_classes,1/6)
        self.assertEqual(len(report['disagreements']),2)
        self.assertEqual(len(report['confusion']),72)
        self.assertTrue(pd.isna(report['per_class'].iloc[0].precision))

    def test_holdout_blanks_overlap_freeze_and_changes(self):
        hold=pd.concat([self.df,self.df.assign(review_id=[str(i) for i in range(100,200)])],ignore_index=True)
        with self.assertRaises(ValueError):j.freeze_holdout(self.spec,'jev-1.14.0','v1',self.df,hold,self.p)
        hold['review_id']=[str(i) for i in range(100,300)]
        j.freeze_holdout(self.spec,'jev-1.14.0','v1',self.df,hold,self.p)
        j.check_frozen(self.p,self.spec,'jev-1.14.0','v1',self.df,hold)
        with self.assertRaises(FileExistsError):j.freeze_holdout(self.spec,'jev-1.14.0','v1',self.df,hold,self.p)
        with self.assertRaises(ValueError):j.check_frozen(self.p,self.spec,'jev-1.14.0','v2',self.df,hold)
        a=j.contract_for(hold,self.spec,'jev-1.14.0','v1','holdout')
        hold.loc[0,j.QUALIFIERS[0]]=1
        b=j.contract_for(hold,self.spec,'jev-1.14.0','v1','holdout')
        self.assertNotEqual(a['human_reference_hash'],b['human_reference_hash'])
        hold.loc[0,j.QUALIFIERS[0]]=None
        with self.assertRaises(ValueError):j.human_check(hold)

if __name__=='__main__':unittest.main()
