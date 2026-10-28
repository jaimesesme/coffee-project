"""Offline evaluation and explicitly invoked Jev pilot/holdout calls."""
import json
import math
from pathlib import Path
import time
import uuid

import pandas as pd
from labeling_runner import (QUALIFIERS, SCHEMA, validate_labels, digest, canonical,
                             read_jsonl, append_json, file_lock, utcnow)

CHOICE_LABELS = {'positive': 1, 'negative': -1, 'unclear': 0}


def human_check(df, expected=None):
    required = ['review_id', 'business_id', 'text'] + QUALIFIERS
    if not set(required).issubset(df.columns):
        raise ValueError('Human file must include IDs, text, and all eight qualifier columns.')
    if df['review_id'].isna().any() or not df['review_id'].is_unique:
        raise ValueError('Missing or duplicate human review IDs.')
    if not df[QUALIFIERS].isin([-1, 0, 1]).all().all():
        raise ValueError('Finish human labels first: every qualifier must be -1, 0, or 1; blanks are unreviewed, not zero.')
    if expected is not None:
        if set(df.review_id) != set(expected.review_id):
            raise ValueError('Human labels do not match the expected review IDs.')
        a = df.set_index('review_id').loc[expected.review_id]
        for col in ['text', 'business_id']:
            if list(a[col]) != list(expected[col]):
                raise ValueError(f'Human reference {col} differs from source.')
    return df[required].copy()


def dataset_hash(df):
    rows = df[['review_id', 'business_id', 'text']].sort_values('review_id').to_dict('records')
    return digest(canonical(rows))


def spec_check(specs):
    if set(specs) != set(QUALIFIERS):
        raise ValueError('Questions must have exactly the eight feature names.')
    for q in specs.values():
        if q.get('type') != 'choice' or set(q.get('criteria', {})) != set(CHOICE_LABELS) or not q.get('instructions'):
            raise ValueError('Each question must be a Choice with positive, negative, unclear criteria.')


def contract_for(df, specs, model, version, stage):
    spec_check(specs)
    contract = {'requested_model': model, 'prompt_version': version,
                'prompt_hash': digest(canonical(specs)), 'schema_hash': digest(canonical(SCHEMA)),
                'dataset_hash': dataset_hash(df), 'stage': stage}
    if stage == 'holdout':
        human_check(df)
        reference = df[['review_id'] + QUALIFIERS].sort_values('review_id').copy()
        reference[QUALIFIERS] = reference[QUALIFIERS].astype(int)
        contract['human_reference_hash'] = digest(canonical(reference.to_dict('records')))
    return contract


def predictions(path, contract=None):
    rows = read_jsonl(path)
    seen = set()
    for r in rows:
        if r['review_id'] in seen:
            raise ValueError('Duplicate saved prediction ID.')
        seen.add(r['review_id'])
        validate_labels(r['labels'])
        if contract and any(r.get(k) != v for k, v in contract.items()):
            raise ValueError('Saved Jev results are incompatible. Restore settings or choose a new version/output file.')
    return rows


def parse_answers(response):
    raw = response.model_dump()
    answers = raw.get('answers', {})
    if set(answers) != set(QUALIFIERS):
        raise ValueError('Jev did not return exactly eight answers.')
    labels = {}
    for q, answer in answers.items():
        if answer.get('type') != 'choice' or answer.get('choice') not in CHOICE_LABELS:
            raise ValueError(f'Invalid choice for {q}.')
        conf = answer.get('confidence')
        probs = answer.get('probabilities', {})
        if not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('Invalid confidence.')
        if set(probs) != set(CHOICE_LABELS) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError('Invalid probabilities.')
        if not math.isclose(sum(probs.values()), 1, abs_tol=.02):
            raise ValueError('Probabilities do not sum approximately to one.')
        labels[q] = CHOICE_LABELS[answer['choice']]
    return validate_labels(labels), raw


def run_jev(df, specs, client, model, version, path, stage='pilot', min_interval=.5, limit=100):
    """Resumable sequential labeling; full runs use stage='full', limit=None."""
    if stage not in ('pilot', 'holdout', 'full'):
        raise ValueError('Stage must be pilot, holdout, or full.')
    if stage != 'full' and len(df) != (100 if stage == 'pilot' else 200):
        raise ValueError('Expected 100 pilot or 200 holdout reviews.')
    if df.empty or not df.review_id.is_unique:
        raise ValueError('Source must be nonempty with unique review IDs.')
    for column in ['review_id', 'business_id', 'text']:
        if not df[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'Source requires nonempty strings in {column}.')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Limit must be a positive integer or None for all reviews.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = contract_for(df, specs, model, version, stage)
    fail_path = path.with_suffix('.failures.jsonl')
    session_path = path.with_suffix('.sessions.jsonl')
    with file_lock(path.with_suffix('.lock')):
        saved = predictions(path, contract)
        config_path = path.with_suffix('.config.json')
        config = {'contract': contract, 'questions': specs}
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Saved experiment configuration differs; choose a new output/version.')
        else:
            with config_path.open('x') as file:
                json.dump(config, file, indent=2)
        done = {r['review_id'] for r in saved}
        if not done.issubset(set(df.review_id)):
            raise ValueError('Saved predictions contain review IDs outside this dataset.')
        pending = df.loc[~df.review_id.isin(done)]
        if limit is not None:
            pending = pending.head(limit)
        print(f'{stage}: {len(done)}/{len(df)} already saved; {len(pending)} requests pending.')
        # Raw question dictionaries are supported by the TypeSafe SDK.
        session_id = uuid.uuid4().hex
        started = time.perf_counter()
        attempted = successful = failed = 0
        try:
            for row in pending.to_dict('records'):
                raw = {}
                call_start = time.perf_counter()
                record = None
                error = None
                try:
                    # Only text is sent, never the human labels or review metadata.
                    response = client.system_one(state=row['text'], questions=specs)
                    raw = response.model_dump()
                    labels, raw = parse_answers(response)
                    if raw.get('model') != model:
                        raise ValueError('Returned model differs from pinned model; do not mix releases.')
                    record = {**contract, 'review_id': row['review_id'], 'business_id': row['business_id'],
                              'text_hash': digest(row['text']), 'labels': labels, 'label_model': raw['model'],
                              'label_date': utcnow(), 'request_id': getattr(response, 'request_id', None),
                              'session_id': session_id, 'answers': raw['answers']}
                except Exception as exc:
                    error = {'error_type': type(exc).__name__, 'error': str(exc),
                             'status_code': getattr(exc, 'status_code', None)}
                elapsed = time.perf_counter() - call_start
                usage = raw.get('usage') or {}
                metadata = {'request_seconds': elapsed, 'input_tokens': usage.get('input_tokens'),
                            'output_tokens': usage.get('output_tokens')}
                attempted += 1
                if record:
                    append_json(path, {**record, **metadata})
                    done.add(row['review_id']); successful += 1
                else:
                    append_json(fail_path, {**contract, 'review_id': row['review_id'],
                                           'business_id': row['business_id'], 'session_id': session_id,
                                           'label_date': utcnow(), **metadata, **error})
                    failed += 1
                if stage != 'full' or error or attempted % 100 == 0 or attempted == len(pending):
                    elapsed_session = time.perf_counter() - started
                    print(f'{stage}: {len(done)}/{len(df)} saved; this session failures={failed}; '
                          f'elapsed={elapsed_session / 60:.1f} min')
                if error and error['status_code'] in (401, 403, 429):
                    print('Stopped for authorization/rate limit. Inspect error, wait or fix settings, then resume.')
                    break
                time.sleep(max(0, min_interval - elapsed))
        finally:
            append_json(session_path, {**contract, 'session_id': session_id, 'label_date': utcnow(),
                                      'wall_seconds': time.perf_counter() - started,
                                      'attempted': attempted, 'successful': successful, 'failed': failed,
                                      'min_interval': min_interval, 'concurrency': 1})
    return predictions(path, contract)


def timing_cost(path, input_price=.042, output_price=0):
    """Measured time and token-based estimate, not vendor benchmark or invoice."""
    path = Path(path)
    records = predictions(path)
    failures = read_jsonl(path.with_suffix('.failures.jsonl'))
    sessions = read_jsonl(path.with_suffix('.sessions.jsonl'))
    attempts = records + failures
    times = pd.Series([r['request_seconds'] for r in attempts], dtype=float)
    wall = sum(s['wall_seconds'] for s in sessions)
    known = [r for r in attempts if r.get('input_tokens') is not None and r.get('output_tokens') is not None]
    inp = sum(r['input_tokens'] for r in known)
    out = sum(r['output_tokens'] for r in known)
    return {'successful':len(records), 'failed_attempts':len(failures), 'wall_seconds_recorded':wall,
            'median_request_seconds':times.median() if len(times) else None,
            'p95_request_seconds':times.quantile(.95) if len(times) else None,
            'successful_reviews_per_minute':len(records)/wall*60 if wall else None,
            'input_tokens_known':inp, 'output_tokens_known':out,
            'estimated_usd_known_usage':(inp*input_price+out*output_price)/1e6,
            'attempts_without_usage':len(attempts)-len(known),
            'input_usd_per_million':input_price, 'output_usd_per_million':output_price}


def evaluate(human, rows, require_complete=True):
    human = human_check(human)
    ids = [r['review_id'] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(human.review_id)):
        raise ValueError('Prediction IDs duplicated or outside human reference.')
    if require_complete and set(ids) != set(human.review_id):
        raise ValueError(f'Only {len(ids)}/{len(human)} reviews predicted. Finish/retry before final evaluation.')
    by_id = {}
    for r in rows:
        labels = r.get('labels', {q:r.get(q) for q in QUALIFIERS})
        validate_labels({q:labels[q] for q in QUALIFIERS})  # GPT historical social_setting excluded.
        by_id[r['review_id']] = labels
    truth = human.set_index('review_id')
    summary, classes, confusion, differences = [], [], [], []
    for q in QUALIFIERS:
        cm = {(a,b):0 for a in [-1,0,1] for b in [-1,0,1]}
        for rid, labels in by_id.items():
            actual, pred = int(truth.loc[rid,q]), labels[q]
            cm[actual,pred] += 1
            if actual != pred:
                differences.append({'review_id':rid,'qualifier':q,'human_label':actual,
                                    'model_label':pred,'text':truth.loc[rid,'text']})
        scores=[]
        for label in [-1,0,1]:
            tp=cm[label,label];support=sum(cm[label,b] for b in [-1,0,1]);predicted=sum(cm[a,label] for a in [-1,0,1])
            precision=tp/predicted if predicted else None
            recall=tp/support if support else None
            f1=2*tp/(support+predicted) if support+predicted else None
            if f1 is not None:scores.append(f1)
            classes.append({'qualifier':q,'label':label,'precision':precision,'recall':recall,'f1':f1,
                            'human_count':support,'predicted_count':predicted})
        n=len(by_id);match=sum(cm[a,a] for a in [-1,0,1])
        summary.append({'qualifier':q,'evaluated_reviews':n,'disagreements':n-match,
                        'agreement_pct':100*match/n if n else None,
                        'macro_f1_present_classes':sum(scores)/len(scores) if scores else None})
        confusion.extend({'qualifier':q,'human_label':a,'predicted_label':b,'count':v} for (a,b),v in cm.items())
    return {'summary':pd.DataFrame(summary),'per_class':pd.DataFrame(classes),'confusion':pd.DataFrame(confusion),
            'disagreements':pd.DataFrame(differences,columns=['review_id','qualifier','human_label','model_label','text'])}


def save_reports(report, folder):
    folder=Path(folder)/('evaluation_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True)
    for name,df in report.items():df.to_csv(folder/(name+'.csv'),index=False)
    print('Saved report:',folder)
    return folder


def freeze_holdout(specs, model, version, pilot, holdout, path):
    spec_check(specs)
    if len(pilot)!=100 or len(holdout)!=200 or set(pilot.review_id)&set(holdout.review_id):
        raise ValueError('Require separate 100-review pilot and 200-review holdout.')
    record={'questions':specs,'requested_model':model,'prompt_version':version,
            'prompt_hash':digest(canonical(specs)),'pilot_hash':dataset_hash(pilot),
            'holdout_hash':dataset_hash(holdout),'frozen_at':utcnow()}
    with Path(path).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    return record


def check_frozen(path, specs, model, version, pilot, holdout):
    f=json.loads(Path(path).read_text())
    expected={'requested_model':model,'prompt_version':version,'prompt_hash':digest(canonical(specs)),
              'pilot_hash':dataset_hash(pilot),'holdout_hash':dataset_hash(holdout)}
    if any(f[k]!=v for k,v in expected.items()):
        raise ValueError('Frozen evaluation settings/data changed. Do not tune against this holdout; use a new independent test set if necessary.')
    return f
