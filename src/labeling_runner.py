"""Resumable Yelp labeling. Importing this module never contacts OpenAI."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict

MODEL = 'gpt-4.1-mini'


class ReviewLabels(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    good_wifi: Literal[-1, 0, 1]
    easy_parking: Literal[-1, 0, 1]
    quiet_atmosphere: Literal[-1, 0, 1]
    enough_seating: Literal[-1, 0, 1]
    outlets_available: Literal[-1, 0, 1]
    work_friendly: Literal[-1, 0, 1]
    good_drinks: Literal[-1, 0, 1]
    fun_atmosphere: Literal[-1, 0, 1]


QUALIFIERS = list(ReviewLabels.model_fields)
SCHEMA = ReviewLabels.model_json_schema()
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def validate_labels(labels):
    # Literal alone can accept bools / numerically equivalent floats: reject them.
    if not isinstance(labels, dict) or set(labels) != set(QUALIFIERS):
        raise ValueError('Label schema must contain exactly the eight ReviewLabels fields.')
    if any(type(v) is not int or v not in (-1, 0, 1) for v in labels.values()):
        raise ValueError('Labels must be integer -1, 0, or 1 (not strings, floats, or booleans).')
    return ReviewLabels.model_validate(labels).model_dump()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def file_lock(path):
    """OS releases the lock even when the kernel exits; never delete lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another labeling operation is active. Stop it first.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_jsonl(path, repair_tail=False, validator=None):
    """Never ignore a malformed complete line; repair only an unfinished final line.

    Explicit repair preserves an exact backup first. A valid final record lacking a
    newline is retained and terminated, not discarded. Semantic errors never repair.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, offset, cut = [], 0, None
    for i, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'{path.name}: malformed complete record at line {i + 1}; file unchanged.') from exc
            if not repair_tail:
                raise ValueError(f'{path.name}: incomplete final record. Set REPAIR_INCOMPLETE_TAIL=True to back up and quarantine it.') from exc
            cut = offset
            break
        if validator:
            validator(record)
        records.append(record)
        offset += len(line)
    needs_newline = bool(raw) and not raw.endswith(b'\n')
    if cut is not None or needs_newline:
        if not repair_tail:
            raise ValueError(f'{path.name}: final newline missing. Enable explicit tail repair; valid records will be retained.')
        backup = path.with_name(path.name + '.before_tail_repair.' + uuid.uuid4().hex + '.bak')
        shutil.copy2(path, backup)
        if cut is not None:
            quarantine = backup.with_suffix('.partial')
            quarantine.write_bytes(raw[cut:])
            with path.open('r+b') as f:
                f.truncate(cut)
                f.flush()
                os.fsync(f.fileno())
            print(f'Quarantined incomplete final bytes: {quarantine}; full backup: {backup}')
        else:
            with path.open('ab') as f:
                f.write(b'\n')
                f.flush()
                os.fsync(f.fileno())
            print(f'Retained valid final record and added newline; backup: {backup}')
    return records


def usage_of(body):
    u = body.get('usage') or {}
    return {'input_tokens': u.get('input_tokens'),
            'output_tokens': u.get('output_tokens'),
            'cached_input_tokens': (u.get('input_tokens_details') or {}).get('cached_tokens')}


def parse_body(body):
    if body.get('status') != 'completed':
        raise ValueError(f'Response not completed: {body.get("status")}; {body.get("incomplete_details")}')
    texts = []
    for item in body.get('output', []):
        for content in item.get('content', []):
            if content.get('type') == 'refusal':
                raise ValueError('Model refusal: ' + content.get('refusal', ''))
            if content.get('type') == 'output_text':
                texts.append(content['text'])
    return validate_labels(json.loads(''.join(texts)))


class Run:
    def __init__(self, rows, prompt, version, results_path, batch_root, repair_tail=False):
        self.rows = {r['review_id']: dict(r) for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate source review IDs.')
        for r in rows:
            if not all(isinstance(r.get(k), str) and r[k].strip() for k in ('review_id', 'business_id', 'text')):
                raise ValueError('Source requires nonempty review_id, business_id, and text. Inspect blank rows separately.')
        if not prompt.strip():
            raise ValueError('Prompt is empty.')
        self.prompt = prompt.strip()  # Matches the original sequential notebook.
        self.contract = {'requested_model': MODEL, 'prompt_version': version,
                         'prompt_hash': digest(self.prompt), 'schema_hash': digest(canonical(SCHEMA))}
        self.path = Path(results_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fail_path = self.path.with_suffix('.failures.jsonl')
        self.batch_root = Path(batch_root).resolve()
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.batch_root / '.labeling.lock'
        self.success = {}
        with file_lock(self.lock_path):
            self.refresh(repair_tail)
            config_path = self.path.with_suffix('.config.json')
            if config_path.exists():
                saved_config = json.loads(config_path.read_text())
                if saved_config['contract'] != self.contract or saved_config['prompt'] != self.prompt:
                    raise ValueError('Results configuration differs; select a matching file or start a new file.')
            else:
                atomic_json(config_path, {'contract': self.contract, 'prompt': self.prompt,
                                          'schema': SCHEMA, 'created_at': utcnow()})
        legacy = sum('schema_hash' not in r for r in self.success.values())
        if legacy:
            print(f'Validated {legacy} legacy records by exact eight-field schema and metadata. '
                  'Legacy records have no text hash/usage; keep the source unchanged.')

    def validate_record(self, r):
        for k, expected in self.contract.items():
            if k == 'schema_hash' and k not in r:
                continue  # Legacy schema inferred from exact validated keys/types below.
            if r.get(k) != expected:
                raise ValueError(f'Incompatible results: {k} differs. Select a matching prompt/version/model/schema or a NEW results file.')
        validate_labels(r.get('labels'))
        if not r.get('label_date') or not r.get('label_model'):
            raise ValueError('Missing model information or label_date.')
        if not (r['label_model'] == MODEL or r['label_model'].startswith(MODEL + '-')):
            raise ValueError('Incompatible label_model.')
        source = self.rows.get(r.get('review_id'))
        if not source or source['business_id'] != r.get('business_id'):
            raise ValueError('Saved review/business ID does not match the source dataset.')
        if r.get('text_hash') and r['text_hash'] != digest(source['text']):
            raise ValueError('Review text changed since labeling; use a new results file.')

    def refresh(self, repair_tail=False):
        seen = {}
        def check(r):
            self.validate_record(r)
            if r['review_id'] in seen:
                raise ValueError('Duplicate review ID in successful results; inspect before continuing.')
            seen[r['review_id']] = r
        read_jsonl(self.path, repair_tail, check)
        # Validate failure file too, before appending anything.
        read_jsonl(self.fail_path, repair_tail)
        self.success = seen
        return seen

    def manifests(self):
        return [(p, json.loads(p.read_text())) for p in self.batch_root.glob('*/manifest.json')]

    def reserved(self):
        ids = set()
        for _, m in self.manifests():
            if m['contract'] == self.contract and m['state'] not in ('collected', 'abandoned'):
                ids.update(r['review_id'] for r in m['requests'].values())
        return ids

    def pending(self, limit=50, retry_failed_only=False):
        failures = {r.get('review_id') for r in read_jsonl(self.fail_path)}
        reserved = self.reserved()
        rows = [r for rid, r in self.rows.items() if rid not in self.success and rid not in reserved
                and (not retry_failed_only or rid in failures)]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('Run limit must be a positive integer or None.')
        return rows if limit is None else rows[:limit]

    def counts(self):
        failed = {r.get('review_id') for r in read_jsonl(self.fail_path)} & self.rows.keys() - self.success.keys()
        return {'completed': len(self.success), 'remaining': len(self.rows) - len(self.success),
                'failed_unresolved': len(failed), 'reserved_for_batch': len(self.reserved() - self.success.keys())}

    def record(self, row, body, labels, mode, request_id=None):
        record = {**self.contract, 'review_id': row['review_id'], 'business_id': row['business_id'],
                  'text_hash': digest(row['text']), 'labels': validate_labels(labels),
                  'label_model': body.get('model'), 'label_date': utcnow(),
                  'mode': mode, 'response_id': body.get('id'), 'request_id': request_id,
                  **usage_of(body)}
        self.validate_record(record)
        return record

    def save(self, record):
        if record['review_id'] in self.success:
            return False
        self.validate_record(record)
        append_json(self.path, record)
        self.success[record['review_id']] = record
        return True

    def failure(self, row, error, mode, body=None, **details):
        append_json(self.fail_path, {**self.contract, 'review_id': row['review_id'],
                    'business_id': row['business_id'], 'label_date': utcnow(), 'mode': mode,
                    'error': str(error), 'response_id': (body or {}).get('id'),
                    **usage_of(body or {}), **details})

    def body(self, row, max_output_tokens=512):
        return {'model': MODEL, 'input': [{'role': 'system', 'content': self.prompt},
                                        {'role': 'user', 'content': row['text']}],
                'text': {'format': {'type': 'json_schema', 'name': 'ReviewLabels',
                                    'strict': True, 'schema': SCHEMA}},
                'max_output_tokens': max_output_tokens, 'store': False}


class RateGate:
    """Shared rolling RPM/TPM gate plus a global server-requested cooldown."""
    def __init__(self, rpm, tpm):
        if rpm < 1 or tpm < 1:
            raise ValueError('Set positive RPM/TPM budgets below your account limits.')
        self.rpm, self.tpm = rpm, tpm
        self.events = deque()
        self.lock = asyncio.Lock()
        self.blocked_until = 0

    def cooldown(self, seconds):
        self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)

    async def acquire(self, tokens):
        if tokens > self.tpm:
            raise ValueError('One request exceeds configured TPM budget; raise budget within account limits or inspect this review.')
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()
                wait = max(0, self.blocked_until - now)
                if len(self.events) >= self.rpm or sum(t for _, t in self.events) + tokens > self.tpm:
                    wait = max(wait, 60 - (now - self.events[0][0]))
                if wait <= 0:
                    self.events.append((now, tokens))
                    return
            await asyncio.sleep(wait + 0.01)


def retry_info(exc, attempt):
    status = getattr(exc, 'status_code', None)
    error_body = getattr(exc, 'body', None) or {}
    code = error_body.get('code') or (error_body.get('error') or {}).get('code') if isinstance(error_body, dict) else None
    if code in ('insufficient_quota', 'credit_balance_exhausted'):
        return False, 0, True
    transient = status in (408, 409, 429) or (status is not None and status >= 500) or type(exc).__name__ in ('APIConnectionError', 'APITimeoutError', 'TimeoutError', 'ConnectionError')
    headers = getattr(getattr(exc, 'response', None), 'headers', {})
    wait = min(60, 2 ** attempt) + random.random()
    try:
        if headers.get('retry-after-ms'):
            wait = max(wait, float(headers['retry-after-ms']) / 1000)
        elif headers.get('retry-after'):
            try:
                seconds = float(headers['retry-after'])
            except ValueError:
                seconds = parsedate_to_datetime(headers['retry-after']).timestamp() - time.time()
            wait = max(wait, seconds)
    except (ValueError, TypeError):
        pass
    return transient, wait, status in (401, 403)


async def run_concurrent(run, client, *, limit=50, concurrency=5, rpm=60, tpm=30000,
                         max_retries=4, max_output_tokens=512, retry_failed_only=False):
    """Workers do API calls only; the coordinator is the sole results writer."""
    if not 1 <= concurrency <= 5 or not 0 <= max_retries <= 10:
        raise ValueError('Concurrency must be 1–5; max_retries must be 0–10.')
    with file_lock(run.lock_path):
        run.refresh()
        rows = run.pending(limit, retry_failed_only)
        print('Starting:', run.counts(), 'selected:', len(rows))
        gate = RateGate(rpm, tpm)
        started = time.monotonic()
        iterator = iter(rows)
        stopped = False
        finished = 0
        ok = 0
        last_print = 0.0

        async def one(row):
            body = None
            for attempt in range(max_retries + 1):
                try:
                    # UTF-8 bytes + schema + framing is deliberately conservative,
                    # not a billing estimate. Reserves the output cap as well.
                    tokens = len(canonical(run.body(row, max_output_tokens)).encode()) + max_output_tokens + 256
                    await gate.acquire(tokens)
                    rsp = await client.responses.create(**run.body(row, max_output_tokens))
                    body = rsp.model_dump()
                    record = run.record(row, body, parse_body(body), 'concurrent', getattr(rsp, '_request_id', None))
                    return row, record, None, body, attempt + 1, False
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    transient, delay, fatal = retry_info(exc, attempt)
                    if transient:
                        gate.cooldown(delay)
                    if transient and attempt < max_retries:
                        continue
                    return row, None, exc, body, attempt + 1, fatal

        tasks = set()
        def launch():
            try:
                row = next(iterator)
            except StopIteration:
                return False
            tasks.add(asyncio.create_task(one(row)))
            return True

        for _ in range(min(concurrency, len(rows))):
            launch()
        try:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    row, record, error, body, attempts, fatal = task.result()
                    if record:
                        run.save(record)
                        ok += 1
                    else:
                        run.failure(row, error, 'concurrent', body, attempts=attempts,
                                    status_code=getattr(error, 'status_code', None),
                                    request_id=getattr(error, 'request_id', None))
                    stopped |= fatal
                    finished += 1
                elapsed = time.monotonic() - started
                speed = finished / elapsed if elapsed else 0
                eta = (len(rows) - finished) / speed if speed else None
                if elapsed - last_print >= 5 or finished == len(rows) or stopped:
                    print(f'{run.counts()} | selected {finished}/{len(rows)} | elapsed {elapsed:.1f}s '
                          f'| {speed * 60:.1f} reviews-finished/min | selected ETA {eta:.0f}s')
                    last_print = elapsed
                if not stopped:
                    while len(tasks) < concurrency and launch():
                        pass
            if stopped:
                print('Stopped scheduling: authentication/billing failure. Fix it before retrying.')
        finally:
            # Interruption keeps responses already completed, then cancels in-flight work.
            for task in tasks:
                if task.done() and not task.cancelled():
                    row, record, error, body, attempts, _ = task.result()
                    if record:
                        run.save(record)
                    else:
                        run.failure(row, error, 'concurrent', body, attempts=attempts)
                else:
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return run.counts()


def prepare_batch(run, *, limit=50, retry_failed_only=False, max_output_tokens=512):
    """Local only. Reserve selected IDs so another mode won't duplicate them."""
    with file_lock(run.lock_path):
        run.refresh()
        rows = run.pending(limit, retry_failed_only)
        if not rows:
            raise ValueError('No unreserved pending reviews. Collect existing batches or inspect counts.')
        if len(rows) > 50000:
            raise ValueError('Batch limit is 50,000 requests. Select a smaller run limit.')
        folder = run.batch_root / uuid.uuid4().hex
        folder.mkdir()
        requests, lines = {}, []
        for row in rows:
            cid = 'review-' + digest(row['review_id'])
            requests[cid] = {'review_id': row['review_id'], 'business_id': row['business_id'],
                             'text_hash': digest(row['text'])}
            lines.append(json.dumps({'custom_id': cid, 'method': 'POST', 'url': '/v1/responses',
                                     'body': run.body(row, max_output_tokens)}, ensure_ascii=False) + '\n')
        payload = ''.join(lines).encode('utf-8')
        if len(payload) >= 200_000_000:
            raise ValueError('Batch input reaches 200 MB. Lower RUN_LIMIT and prepare smaller batches.')
        input_path = folder / 'input.jsonl'
        input_path.write_bytes(payload)
        manifest = {'state': 'prepared', 'contract': run.contract, 'results_path': str(run.path),
                    'input_path': str(input_path), 'input_hash': hashlib.sha256(payload).hexdigest(),
                    'created_at': utcnow(), 'requests': requests, 'batch_id': None}
        atomic_json(folder / 'manifest.json', manifest)
        print(f'Prepared {len(rows)} requests ({len(payload):,} bytes). No API calls. Manifest: {folder / "manifest.json"}')
        return folder / 'manifest.json'


def checked_manifest(run, path):
    path = Path(path)
    m = json.loads(path.read_text())
    if m['contract'] != run.contract or m['results_path'] != str(run.path):
        raise ValueError('Batch manifest does not match this prompt/schema/model/version/results file.')
    for req in m['requests'].values():
        row = run.rows.get(req['review_id'])
        if not row or row['business_id'] != req['business_id'] or digest(row['text']) != req['text_hash']:
            raise ValueError('Batch source data changed; use the original source.')
    return path, m


def submit_batch(run, client, manifest_path):
    with file_lock(run.lock_path):
        run.refresh()
        path, m = checked_manifest(run, manifest_path)
        if m['state'] != 'prepared' or m.get('batch_id'):
            raise ValueError('This manifest was already submitted or submission is uncertain. Use status/recovery, NEVER resubmit it.')
        if any(x['review_id'] in run.success for x in m['requests'].values()):
            raise ValueError('Some prepared reviews are now completed. Abandon this unsubmitted manifest and prepare again.')
        raw = Path(m['input_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != m['input_hash']:
            raise ValueError('Batch input was edited after preparation.')
        m['state'] = 'submitting'
        atomic_json(path, m)  # Before ANY remote operation: a crash cannot trigger resubmission.
        try:
            with Path(m['input_path']).open('rb') as f:
                uploaded = client.files.create(file=f, purpose='batch')
            m['input_file_id'] = uploaded.id
            atomic_json(path, m)
            batch = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/responses',
                                          completion_window='24h', metadata={'local_manifest': path.parent.name})
            m.update(state='submitted', batch_id=batch.id, batch=batch.model_dump())
            atomic_json(path, m)
        except BaseException:
            print('Submission interrupted/uncertain. Do not resubmit. Run recover_batch to locate the server batch.')
            raise
        print('Submitted batch:', m['batch_id'])
        return m['batch_id']


def recover_batch(run, client, manifest_path):
    """Read-only server search after a restart or uncertain submission; never creates."""
    with file_lock(run.lock_path):
        path, m = checked_manifest(run, manifest_path)
        if m.get('batch_id'):
            return m['batch_id']
        matches = [b for b in client.batches.list(limit=100)
                   if (b.metadata or {}).get('local_manifest') == path.parent.name
                   or (m.get('input_file_id') and b.input_file_id == m['input_file_id'])]
        if len(matches) != 1:
            raise ValueError(f'Found {len(matches)} matching batches. Inspect the dashboard/files; manifest stays blocked to prevent duplicate billing.')
        b = matches[0]
        m.update(state='submitted', batch_id=b.id, batch=b.model_dump())
        atomic_json(path, m)
        return b.id


def abandon_prepared(run, manifest_path):
    with file_lock(run.lock_path):
        path, m = checked_manifest(run, manifest_path)
        if m['state'] != 'prepared':
            raise ValueError('Only a never-submitted prepared manifest can be abandoned locally.')
        m['state'] = 'abandoned'
        atomic_json(path, m)


def batch_status(run, client, manifest_path):
    with file_lock(run.lock_path):
        path, m = checked_manifest(run, manifest_path)
        if not m.get('batch_id'):
            raise ValueError('No batch ID. Submit a prepared manifest or recover an uncertain submission.')
        batch = client.batches.retrieve(m['batch_id'])
        m['batch'] = batch.model_dump()
        if m['state'] != 'collected':
            m['state'] = batch.status
        atomic_json(path, m)
        print('Batch:', batch.id, 'status:', batch.status, 'counts:', batch.request_counts)
        return batch


def collect_batch(run, client, manifest_path):
    """Collect terminal batches, including successful portions of expired batches."""
    with file_lock(run.lock_path):
        run.refresh()
        path, m = checked_manifest(run, manifest_path)
        if not m.get('batch_id'):
            raise ValueError('No batch ID. Recover submission first.')
        batch = client.batches.retrieve(m['batch_id'])
        m['batch'] = batch.model_dump()
        atomic_json(path, m)
        if batch.status not in TERMINAL:
            print('Batch is still', batch.status, '— check again later. Nothing imported.')
            return
        if m['state'] == 'collected':
            print('Already collected. No duplicate successes appended.')
            return m.get('report')
        entries = []
        for kind, fid in [('output', batch.output_file_id), ('errors', batch.error_file_id)]:
            if not fid:
                continue
            target = path.parent / f'{kind}_{fid}.jsonl'
            if not target.exists():
                data = client.files.content(fid).content
                temp = target.with_suffix('.download')
                temp.write_bytes(data)
                os.replace(temp, target)
            entries.extend(read_jsonl(target))
        # Check all mapping identities before saving any result.
        grouped = {}
        for e in entries:
            cid = e.get('custom_id')
            if cid not in m['requests']:
                raise ValueError(f'Unexpected custom_id {cid}; downloaded files retained for inspection.')
            grouped.setdefault(cid, []).append(e)
        report = {'saved': 0, 'already_saved': 0, 'failed': 0, 'missing': 0, 'expired': 0}
        for cid, req in m['requests'].items():
            row = run.rows[req['review_id']]
            if row['review_id'] in run.success:
                report['already_saved'] += 1
                continue
            matches = grouped.get(cid, [])
            body = None
            try:
                if not matches:
                    report['missing'] += 1
                    raise ValueError(f'Missing result; batch status={batch.status}; errors={getattr(batch, "errors", None)}')
                if len(matches) != 1:
                    raise ValueError('Duplicate custom_id in downloaded results; inspect raw files.')
                entry = matches[0]
                if entry.get('error'):
                    if entry['error'].get('code') == 'batch_expired':
                        report['expired'] += 1
                    raise ValueError(canonical(entry['error']))
                response = entry.get('response') or {}
                body = response.get('body') or {}
                if response.get('status_code') != 200:
                    raise ValueError(f'HTTP {response.get("status_code")}: {body.get("error")}')
                record = run.record(row, body, parse_body(body), 'batch', response.get('request_id'))
                record['batch_id'] = batch.id
                record['custom_id'] = cid
                run.save(record)
                report['saved'] += 1
            except Exception as exc:
                run.failure(row, exc, 'batch', body, custom_id=cid, batch_id=batch.id, batch_status=batch.status)
                report['failed'] += 1
        m.update(state='collected', collected_at=utcnow(), report=report)
        atomic_json(path, m)
        print('Collected:', report, 'Overall:', run.counts())
        return report


def usage_report(run, pricing):
    """Cost is an estimate from available usage, never a fixed per-review price."""
    run.refresh()
    records = list(run.success.values()) + read_jsonl(run.fail_path)
    totals = {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0,
              'estimated_usd': 0.0, 'records_without_usage': 0}
    seen_responses = set()
    for r in records:
        usage_key = r.get('response_id') or ((r.get('batch_id'), r.get('custom_id')) if r.get('batch_id') else None)
        if usage_key and usage_key in seen_responses:
            continue
        if usage_key:
            seen_responses.add(usage_key)
        if r.get('input_tokens') is None or r.get('output_tokens') is None:
            totals['records_without_usage'] += 1
            continue
        inp, out = r['input_tokens'], r['output_tokens']
        cached = min(inp, r.get('cached_input_tokens') or 0)
        rates = pricing[r.get('mode', 'concurrent')]
        totals['input_tokens'] += inp
        totals['cached_input_tokens'] += cached
        totals['output_tokens'] += out
        totals['estimated_usd'] += ((inp - cached) * rates['input'] + cached * rates['cached_input'] + out * rates['output']) / 1_000_000
    print('Estimated cost for recorded usage only (USD; not an invoice):', totals)
    print('Excludes unknown usage from legacy records, interrupted requests, and retries without usage. Cached tokens are part of input, not extra input.')
    return totals
