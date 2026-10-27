# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate retained CPU-only reference evidence and freeze its checksums."""
import hashlib,json,re,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=runpy.run_path(str(ROOT/'probe.py'));v=runpy.run_path(str(ROOT/'verify.py'))
rows=json.loads((ROOT/'score-cases-v2.json').read_text())['cases']
assert len(rows)==28 and len({r['name'] for r in rows})==28
for row in rows:
    assert row['equations_match']
    expected=v['expected'](row['q'],row['d'])
    assert expected==row['expected'],row['name']
    assert row['observed']['cp']==expected['cp'] and row['observed']['wdl']==expected['wdl'],row['name']
sessions=sorted(ROOT.glob('*/argv.json'))
assert len(sessions)==72,len(sessions)
backends=set()
for argv_path in sessions:
    directory=argv_path.parent
    argv=json.loads(argv_path.read_text())
    selected=[x.removeprefix('--backend=') for x in argv if x.startswith('--backend=')]
    assert len(selected)==1 and selected[0] in ('random','trivial','recordreplay')
    backends.update(selected)
    assert '--config=' in argv and '--weights=' in argv and '--threads=1' in argv
    if selected[0]=='recordreplay':
        assert any(x.startswith('--backend-opts=backend=random,random(') for x in argv)
    assert json.loads((directory/'exit.json').read_text())['exit']==0
    initialized=re.findall(r'Creating backend \[([^]]+)\]',(directory/'stderr.txt').read_text())
    # The legacy wrapper logs its CPU child; direct trivial does not log a
    # Creating-backend line in this binary. Both keep the strict argv allowlist.
    assert set(initialized)<=set(('random','trivial','recordreplay')),initialized
    if selected[0]=='recordreplay': assert initialized==['random'],initialized
exe_hash=hashlib.sha256(p['EXE'].read_bytes()).hexdigest()
assert exe_hash==p['EXE_SHA']
audit={'schema':1,'status':'passed','exact_qd_cases':28,'repeated_per_case':2,
       'official_cpu_search_sessions':len(sessions),'all_exit_zero':True,
       'backend_allowlist':sorted(backends),'explicit_cpu_recordreplay_children':True,
       'equations_match_all_cp_and_integer_wdl':True,'official_executable_sha256':exe_hash,
       'product_source_edits':False,'gpu_backend_executions':0}
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
manifest={}
for path in sorted(ROOT.rglob('*')):
    if path.is_file() and path.name!='evidence-manifest.json':
        assert path.resolve().is_relative_to(ROOT)
        manifest[path.relative_to(ROOT).as_posix()]={'bytes':path.stat().st_size,
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'evidence-manifest.json').write_text(json.dumps({'schema':1,'files':manifest},indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
print('Manifest files',len(manifest),'SHA256',hashlib.sha256((ROOT/'evidence-manifest.json').read_bytes()).hexdigest())
for name in ('probe.py','verify.py','extra.py','threshold.py','audit.py','score-cases-v2.json','score-controls-v1.json','source-manifest.json','audit.json'):
    print(name,manifest[name]['sha256'])
