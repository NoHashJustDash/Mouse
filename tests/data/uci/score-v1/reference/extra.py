# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU score option controls, mate priority, defaults and provenance."""
import hashlib,json,runpy,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=runpy.run_path(str(ROOT/'probe.py'));v=runpy.run_path(str(ROOT/'verify.py'))
run=p['run_session']; observed=v['observed']
rows=[]
for name,model,score in [('mu_explicit','mu_pos','WDL_mu'),('old_centipawn','mu_pos','centipawn'),('raw_wl','mu_pos','W-L'),('half_old','half_pos','centipawn')]:
    opts="backend=random,random(seed=0),replay_file='"+(ROOT/(model+'.replay.bin')).as_posix()+"'"
    a=observed(run('control-'+name+'-a','recordreplay',opts,score_type=score))
    b=observed(run('control-'+name+'-b','recordreplay',opts,score_type=score))
    if a!=b: raise AssertionError((name,a,b))
    rows.append({'name':name,'score_type':score,'observed':a})
    print(name,a)
for name,fen,go in [
 ('mate_white','7k/5K2/6Q1/8/8/8/8/8 w - - 0 1','go nodes 2 searchmoves g6g7'),
 ('mate_black','8/8/8/8/8/6q1/5k2/7K b - - 0 1','go nodes 2 searchmoves g3g2'),
 ('trivial_start',p['FEN'],'go nodes 1')]:
    a=run('terminal-'+name+'-a','trivial','',fen=fen,go=go)
    b=run('terminal-'+name+'-b','trivial','',fen=fen,go=go)
    # Ignore timing/node-rate fields; score/PV payload is retained verbatim.
    left=[line[line.index(' score '):] for line in a['info']]
    right=[line[line.index(' score '):] for line in b['info']]
    if left!=right or a['bestmove']!=b['bestmove']: raise AssertionError((name,a,b))
    rows.append({'name':name,'fen':fen,'go':go,'score_pv':left,'bestmove':a['bestmove']})
    print(name,left,a['bestmove'])
opts="backend=random,random(seed=0),replay_file='"+(ROOT/'mu_pos.replay.bin').as_posix()+"'"
no_wdl=run('default-show-wdl','recordreplay',opts,show_wdl=False)
if any(' wdl ' in line for line in no_wdl['info']): raise AssertionError('Default unexpectedly shows WDL')
rows.append({'name':'default_show_wdl_false','info':no_wdl['info']})
help_result=subprocess.run([str(p['EXE']),'--backend=trivial','--config=','--show-hidden','--help'],cwd=ROOT,capture_output=True,timeout=10)
(ROOT/'help-hidden.stdout').write_bytes(help_result.stdout);(ROOT/'help-hidden.stderr').write_bytes(help_result.stderr)
if help_result.returncode: raise AssertionError(help_result.returncode)
(ROOT/'score-controls-v1.json').write_text(json.dumps({'schema':1,'controls':rows},indent=2)+'\n',encoding='utf-8')
pin='fd71a2d921b689c5f479d3227c3806c8e272d9c5'
paths=['src/search/classic/search.cc','src/search/classic/params.cc','src/search/classic/params.h','src/chess/callbacks.h','src/chess/uciloop.cc','src/utils/fastmath.h','src/neural/backends/network_trivial.cc','src/neural/backends/network_random.cc','src/neural/backends/network_record.cc']
manifest=[]
for path in paths:
    url=f'https://raw.githubusercontent.com/LeelaChessZero/lc0/{pin}/{path}'
    raw=urllib.request.urlopen(url,timeout=20).read()
    manifest.append({'path':path,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
(ROOT/'source-manifest.json').write_text(json.dumps({'commit':pin,'files':manifest},indent=2)+'\n',encoding='utf-8')
print('controls',len(rows),'source manifest',len(manifest))
