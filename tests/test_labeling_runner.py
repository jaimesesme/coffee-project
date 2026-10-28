"""Offline only: fake SDK clients, temporary files, no API key/network."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import labeling_runner as lr


class Obj(SimpleNamespace):
    def model_dump(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}


def body(labels=None, status='completed'):
    return {'id': 'resp_test', 'model': lr.MODEL + '-2025-04-14', 'status': status,
            'output': [{'type': 'message', 'content': [{'type': 'output_text',
                        'text': json.dumps(labels if labels is not None else {q: 0 for q in lr.QUALIFIERS})}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 20,
                      'input_tokens_details': {'cached_tokens': 40}}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [{'review_id': str(i), 'business_id': 'biz' + str(i), 'text': 'good coffee'} for i in range(8)]
        self.run = lr.Run(self.rows, 'same prompt', 'v1', self.root / 'results.jsonl', self.root / 'batches')

    def record(self, i=0):
        b = body()
        b['id'] = f'resp_{i}'
        return self.run.record(self.rows[i], b, lr.parse_body(b), 'concurrent')

    def test_resume_legacy_and_reject_contract_schema_source_changes(self):
        r = self.record()
        r.pop('schema_hash')
        r.pop('text_hash')
        lr.append_json(self.run.path, r)
        self.run.refresh()
        self.assertEqual(len(self.run.pending()), 7)
        before = self.run.path.read_bytes()
        for key in ('requested_model', 'prompt_version', 'prompt_hash', 'label_model'):
            bad = dict(r, **{key: 'wrong'})
            with self.assertRaises(ValueError): self.run.validate_record(bad)
        for labels in ({**r['labels'], 'social_setting': 0}, {**r['labels'], 'good_wifi': True},
                       {**r['labels'], 'good_wifi': '1'}, {**r['labels'], 'good_wifi': 1.0}):
            with self.assertRaises(ValueError): self.run.validate_record({**r, 'labels': labels})
        with self.assertRaises(ValueError): self.run.validate_record({**r, 'business_id': 'wrong'})
        with self.assertRaises(ValueError): self.run.validate_record({**r, 'schema_hash': 'wrong'})
        with self.assertRaises(ValueError): self.run.validate_record({**r, 'text_hash': 'wrong'})
        self.assertEqual(self.run.path.read_bytes(), before)

    def test_tail_recovery_preserves_good_records_and_backup(self):
        r = self.record()
        raw = (json.dumps(r) + '\n{"review_id":').encode()
        self.run.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'incomplete final'): self.run.refresh()
        self.assertEqual(self.run.path.read_bytes(), raw)
        self.run.refresh(True)
        self.assertEqual(len(self.run.success), 1)
        self.assertTrue(any(p.read_bytes() == raw for p in self.root.glob('*.bak')))
        self.assertEqual(json.loads(self.run.path.read_text()), r)
        self.run.path.write_text(json.dumps(r))
        self.run.refresh(True)
        self.assertEqual(len(self.run.success), 1)
        self.assertTrue(self.run.path.read_bytes().endswith(b'\n'))

    def test_no_repair_of_complete_corruption_or_duplicate(self):
        for raw in ('broken\n', json.dumps(self.record()) + '\n' + json.dumps(self.record()) + '\n'):
            self.run.path.write_text(raw)
            with self.assertRaises(ValueError): self.run.refresh(True)
            self.assertEqual(self.run.path.read_text(), raw)

    def test_schema_and_response_failures(self):
        for b in (body(status='incomplete'), body({'good_wifi': 1}),
                  {**body(), 'output': [{'content': [{'type': 'refusal', 'refusal': 'no'}]}]}):
            with self.assertRaises(ValueError): lr.parse_body(b)

    def test_concurrency_single_writer_retry_and_resume(self):
        state = {'active': 0, 'peak': 0, 'calls': 0}
        async def create(**kwargs):
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            state['calls'] += 1
            try:
                await asyncio.sleep(.001)
                return Obj(**body())
            finally:
                state['active'] -= 1
        client = Obj(responses=Obj(create=create))
        asyncio.run(lr.run_concurrent(self.run, client, concurrency=5, rpm=1000, tpm=1000000))
        self.assertEqual(len(self.run.success), 8)
        self.assertEqual(state['peak'], 5)
        asyncio.run(lr.run_concurrent(self.run, client, concurrency=5, rpm=1000, tpm=1000000))
        self.assertEqual(state['calls'], 8)
        self.assertEqual(len(lr.read_jsonl(self.run.path)), 8)

    def test_transient_retry_bounded_and_invalid_response_logged(self):
        calls = []
        class Transient(Exception): status_code = 429
        async def create(**kwargs):
            calls.append(1)
            raise Transient('rate limited')
        client = Obj(responses=Obj(create=create))
        with patch.object(lr, 'retry_info', return_value=(True, 0, False)):
            asyncio.run(lr.run_concurrent(self.run, client, limit=1, max_retries=2, rpm=1000, tpm=1000000))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.run.success), 0)
        self.assertEqual(self.run.counts()['failed_unresolved'], 1)
        async def invalid(**kwargs): return Obj(**body(status='incomplete'))
        asyncio.run(lr.run_concurrent(self.run, Obj(responses=Obj(create=invalid)), limit=1, retry_failed_only=True))
        self.assertEqual(len(self.run.success), 0)
        self.assertEqual(lr.read_jsonl(self.run.fail_path)[-1]['input_tokens'], 100)

    def test_retry_after_and_quota(self):
        e = Exception('429');e.status_code = 429;e.response = Obj(headers={'retry-after': '120'})
        retry, delay, fatal = lr.retry_info(e, 0)
        self.assertTrue(retry); self.assertGreaterEqual(delay, 120)
        e.body = {'code': 'credit_balance_exhausted'}
        self.assertEqual(lr.retry_info(e, 0), (False, 0, True))
        async def check():
            gate = lr.RateGate(1, 100)
            await gate.acquire(50)
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(gate.acquire(50), timeout=.01)
            with self.assertRaises(ValueError): await gate.acquire(101)
        asyncio.run(check())

    def test_batch_prepare_submission_once_and_recovery(self):
        path = lr.prepare_batch(self.run, limit=2)
        m = json.loads(path.read_text())
        entries = lr.read_jsonl(m['input_path'])
        self.assertEqual(len({e['custom_id'] for e in entries}), 2)
        self.assertEqual(entries[0]['body']['text']['format']['schema'], lr.SCHEMA)
        self.assertEqual(len(self.run.pending()), 6)
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return Obj(id='batch_1', status='validating', metadata={'local_manifest':path.parent.name}, input_file_id='file_1')
        client = Obj(files=Obj(create=lambda **kw: Obj(id='file_1')), batches=Obj(create=create))
        self.assertEqual(lr.submit_batch(self.run, client, path), 'batch_1')
        with self.assertRaises(ValueError): lr.submit_batch(self.run, client, path)
        self.assertEqual(len(calls), 1)
        m = json.loads(path.read_text());m['batch_id']=None;m['state']='submitting';lr.atomic_json(path,m)
        client.batches.list = lambda **kw: [create()]
        self.assertEqual(lr.recover_batch(self.run, client, path), 'batch_1')
        with self.assertRaises(ValueError): lr.abandon_prepared(self.run, path)

    def test_batch_collect_unordered_invalid_expired_missing_idempotent(self):
        path = lr.prepare_batch(self.run, limit=5)
        m=json.loads(path.read_text());m['state']='submitted';m['batch_id']='batch_2';lr.atomic_json(path,m)
        ids=list(m['requests'])
        entries=[{'custom_id':ids[1],'response':{'status_code':200,'body':body()}},
                 {'custom_id':ids[0],'response':{'status_code':200,'body':body()}},
                 {'custom_id':ids[2],'response':{'status_code':200,'body':body({'bad':1})}}]
        errors=[{'custom_id':ids[3],'error':{'code':'batch_expired'}}]
        batch=Obj(id='batch_2',status='expired',output_file_id='out',error_file_id='err')
        client=Obj(batches=Obj(retrieve=lambda bid:batch),files=Obj(content=lambda fid:Obj(content=('\n'.join(json.dumps(x) for x in (entries if fid=='out' else errors))+'\n').encode())))
        report=lr.collect_batch(self.run,client,path)
        self.assertEqual(report,{'saved':2,'already_saved':0,'failed':3,'missing':1,'expired':1})
        self.assertEqual(len(self.run.success),2)
        raw=self.run.path.read_bytes()
        lr.collect_batch(self.run,client,path)
        self.assertEqual(raw,self.run.path.read_bytes())
        self.assertEqual(len(self.run.pending(retry_failed_only=True)),3)

    def test_batch_uncertain_submission_never_resubmits(self):
        path=lr.prepare_batch(self.run,limit=1)
        def crash(**kwargs):raise TimeoutError('network uncertain')
        client=Obj(files=Obj(create=lambda **kw:Obj(id='file_1')),batches=Obj(create=crash))
        with self.assertRaises(TimeoutError):lr.submit_batch(self.run,client,path)
        with self.assertRaises(ValueError):lr.submit_batch(self.run,client,path)

    def test_usage_cached_not_double_counted(self):
        self.run.save(self.record())
        rates={k:{'input':.4,'cached_input':.1,'output':1.6} for k in ['concurrent','batch']}
        result=lr.usage_report(self.run,rates)
        self.assertAlmostEqual(result['estimated_usd'],(60*.4+40*.1+20*1.6)/1e6)
        self.assertEqual(result['input_tokens'],100)

    def test_real_sdk_request_serialization_with_mock_transport(self):
        import httpx
        from openai import AsyncOpenAI, OpenAI
        captured = []
        def handler(request):
            captured.append((request.url.path, request.content))
            if request.url.path.endswith('/responses'):
                payload = json.loads(request.content)
                self.assertEqual(payload['text']['format']['schema'], lr.SCHEMA)
                self.assertEqual(payload['model'], 'gpt-4.1-mini')
                return httpx.Response(200, json={**body(), 'object': 'response', 'created_at': 1})
            if request.url.path.endswith('/files'):
                return httpx.Response(200, json={'id':'file_mock','object':'file','bytes':1,'created_at':1,'filename':'input.jsonl','purpose':'batch'})
            if request.url.path.endswith('/batches'):
                payload = json.loads(request.content)
                self.assertEqual(payload['endpoint'], '/v1/responses')
                return httpx.Response(200, json={'id':'batch_mock','object':'batch','endpoint':'/v1/responses',
                    'input_file_id':'file_mock','completion_window':'24h','status':'validating','created_at':1})
            raise AssertionError('Unexpected mock route')
        async def check():
            async with AsyncOpenAI(api_key='fake-local-only',max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
                await lr.run_concurrent(self.run,client,limit=1)
        asyncio.run(check())
        self.assertEqual(len(self.run.success),1)
        path=lr.prepare_batch(self.run,limit=1)
        with OpenAI(api_key='fake-local-only',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            self.assertEqual(lr.submit_batch(self.run,client,path),'batch_mock')
        self.assertEqual(len(captured),3)

    def test_interruption_preserves_finished_requests(self):
        async def check():
            entered=asyncio.Event()
            call_count=0
            async def create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return Obj(**body())
                entered.set()
                await asyncio.Event().wait()
            client=Obj(responses=Obj(create=create))
            task=asyncio.create_task(lr.run_concurrent(self.run,client,limit=8,concurrency=1))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.run.refresh()
            self.assertEqual(len(self.run.success),1)
            self.assertEqual(len(self.run.pending()),7)
            with lr.file_lock(self.run.lock_path): pass
        asyncio.run(check())

    def test_unknown_batch_id_does_not_save_partial_output(self):
        path=lr.prepare_batch(self.run,limit=2)
        m=json.loads(path.read_text());m.update(state='submitted',batch_id='batch_bad');lr.atomic_json(path,m)
        cid=next(iter(m['requests']))
        lines=[{'custom_id':cid,'response':{'status_code':200,'body':body()}},
               {'custom_id':'unknown','response':{'status_code':200,'body':body()}}]
        client=Obj(batches=Obj(retrieve=lambda _:Obj(id='batch_bad',status='completed',output_file_id='out',error_file_id=None)),
                   files=Obj(content=lambda _:Obj(content=('\n'.join(json.dumps(x) for x in lines)+'\n').encode())))
        with self.assertRaisesRegex(ValueError,'Unexpected custom_id'):lr.collect_batch(self.run,client,path)
        self.assertEqual(len(self.run.success),0)

    def test_locks(self):
        with lr.file_lock(self.run.lock_path):
            with self.assertRaises(RuntimeError):
                with lr.file_lock(self.run.lock_path):pass


if __name__ == '__main__': unittest.main()
