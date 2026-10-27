# SPDX-License-Identifier: GPL-3.0-or-later
"""Bracket the post-rescaling 0.996 fallback threshold; research only."""
import json,runpy,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=runpy.run_path(str(ROOT/'probe.py'));v=runpy.run_path(str(ROOT/'verify.py'))
key,values=p['records'](ROOT/'record-root.bin')[0]
rows=json.loads((ROOT/'score-cases-v1.json').read_text())['cases']
for suffix,draw in [('below_far',.8955),('below',.8959),('boundary',.896),('above',.8961),('above_far',.8965)]:
    name='threshold_'+suffix;q=v['f32'](.1);d=v['f32'](draw)
    payload=values.copy();payload[0]=q;payload[1]=d
    path=ROOT/(name+'.replay.bin');path.write_bytes(struct.pack('<Qi',key,len(payload))+struct.pack('<'+'f'*len(payload),*payload))
    opts="backend=random,random(seed=0),replay_file='"+path.as_posix()+"'"
    a=v['observed'](p['run_session']('case-'+name+'-a','recordreplay',opts));b=v['observed'](p['run_session']('case-'+name+'-b','recordreplay',opts))
    expected=v['expected'](q,d)
    assert a==b
    match=a['cp']==expected['cp'] and a['wdl']==expected['wdl']
    assert match,(name,a,expected)
    rows.append({'name':name,'q':q,'d':d,'q_bits':struct.pack('<f',q).hex(),'d_bits':struct.pack('<f',d).hex(),'expected':expected,'observed':a,'equations_match':match})
    print(name,'reported_abs_q_plus_d',v['add'](abs(expected['reported_q']),expected['reported_d']),'branch',expected['branch'],'cp',a['cp'],'wdl',a['wdl'])
(ROOT/'score-cases-v2.json').write_text(json.dumps({'schema':1,'cases':rows},indent=2)+'\n',encoding='utf-8')
print('SUMMARY',len(rows),'cases x2, all exact')
