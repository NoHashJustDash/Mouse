# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent equations plus exact official-binary replay fixtures; research only."""
from __future__ import annotations
import json
import math
import re
import runpy
import struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent
probe=runpy.run_path(str(ROOT/'probe.py'))
run_session=probe['run_session']; records=probe['records']

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def add(a,b): return f32(a+b)
def mul(a,b): return f32(a*b)
def div(a,b): return f32(a/b)
def fast_ln(x):
    # Derive mantissa/exponent mathematically; preserve each float32 operation
    # in the pinned approximation, including subtraction of the exponent bias.
    mantissa,exponent=math.frexp(f32(x))
    fraction=f32(mantissa*2-1)
    curve=mul(fraction,add(f32(1.3465552),-mul(f32(0.34655523),fraction)))
    binary_log=add(add(curve,-127),exponent+126)
    return mul(f32(math.log(2)),binary_log)
def fast_power2(x):
    x=f32(x)
    if x < -126: return 0.0
    exponent=math.floor(x)
    if x<0 and x==exponent: exponent-=1
    fraction=add(x,-exponent)
    polynomial=add(1,mul(fraction,add(f32(0.6602339),mul(f32(0.33976606),fraction))))
    return f32(math.ldexp(polynomial,exponent))
def logistic(x):
    x=f32(x)
    if x>20: return 1.0
    if x<-20: return 0.0
    denominator=add(1,fast_power2(mul(f32(1.442695040),-x)))
    return div(1,denominator)
def expected(q,d):
    q=f32(q);d=f32(d);mu=0.0
    w=div(add(add(1,q),-d),2);loss=div(add(add(1,-q),-d),2)
    eps=f32(.0001)
    transformed=all(eps<x<add(1,-eps) for x in (w,d,loss))
    if transformed:
        a=fast_ln(add(div(1,loss),-1));b=fast_ln(add(div(1,w),-1))
        denominator=add(a,b)
        width=div(2,denominator)
        mu=div(add(a,-b),denominator)
        w=logistic(div(add(-1,mu),width))
        loss=logistic(div(add(-1,-mu),width))
        q=add(w,-loss);d=max(0.0,add(add(1,-w),-loss))
    fallback=f32(45*math.tan(1.56728071628*q))
    use_mu=mu!=0 and add(abs(q),d)<f32(.996) and (abs(mu)<1 or abs(fallback)<abs(mul(100,mu)))
    score=mul(100,mu) if use_mu else fallback
    win=max(0,math.floor(500*(1+q-d)+.5));loss=max(0,math.floor(500*(1-q-d)+.5))
    draw=1000-win-loss
    if draw<0:
        win=min(1000,max(0,win+int(draw/2)));loss=1000-win;draw=0
    return {'cp':int(score),'wdl':[win,draw,loss],'branch':'mu' if use_mu else 'fallback',
            'rescaled':transformed,'reported_q':q,'reported_d':d,'mu':mu,'score_float':score}

def observed(result):
    if len(result['info'])<1: raise AssertionError('No score info')
    line=result['info'][-1]
    match=re.search(r'score cp (-?\d+) wdl (\d+) (\d+) (\d+)',line)
    if not match: raise AssertionError(line)
    return {'cp':int(match[1]),'wdl':[int(match[i]) for i in range(2,5)]}

if __name__=='__main__':
    base=records(ROOT/'record-root.bin')
    if len(base)!=1: raise AssertionError('One root evaluation required')
    key,values=base[0]
    definitions=[('recorded',values[0],values[1]),('zero',0,0),('draw',0,1),
        ('half_pos',.5,0),('half_neg',-.5,0),('endpoint_win',1,0),('endpoint_loss',-1,0),
        ('tie_pos',.125,0),('tie_neg',-.125,0),('trunc_pos',.04,0),('trunc_neg',-.04,0),
        ('mu_pos',.3,.4),('mu_neg',-.3,.4),('mu_large_pos',.5,.3),('mu_large_neg',-.5,.3),
        ('mu_zero',0,.5),('mu_high_q',.9,.05),('near_switch',.99,.005),
        ('near_switch2',.99,.004),('eps_draw_equal',.99,f32(.0001)),
        ('eps_draw_above',.99,struct.unpack('<f',struct.pack('<I',struct.unpack('<I',struct.pack('<f',.0001))[0]+1))[0]),
        ('near_draw',0,.9998),('narrow_win',.9998,.00011)]
    results=[]
    for name,q,d in definitions:
        q=f32(q);d=f32(d)
        payload=values.copy();payload[0]=q;payload[1]=d
        path=ROOT/(name+'.replay.bin')
        path.write_bytes(struct.pack('<Qi',key,len(payload))+struct.pack('<'+'f'*len(payload),*payload))
        opts="backend=random,random(seed=0),replay_file='"+path.as_posix()+"'"
        expectation=expected(q,d)
        left=observed(run_session('case-'+name+'-a','recordreplay',opts))
        right=observed(run_session('case-'+name+'-b','recordreplay',opts))
        if left!=right: raise AssertionError(('Nonrepeatable',name,left,right))
        row={'name':name,'q':q,'d':d,'q_bits':struct.pack('<f',q).hex(),
             'd_bits':struct.pack('<f',d).hex(),'expected':expectation,'observed':left,
             'equations_match':left['cp']==expectation['cp'] and left['wdl']==expectation['wdl']}
        results.append(row)
        print(name,'Q',q,'D',d,'branch',expectation['branch'],'expected',expectation['cp'],expectation['wdl'],'actual',left,'match',row['equations_match'])
    (ROOT/'score-cases-v1.json').write_text(json.dumps({'schema':1,'cases':results},indent=2)+'\n',encoding='utf-8')
    print('SUMMARY',len(results),'cases x2; exact equations',sum(r['equations_match'] for r in results))
