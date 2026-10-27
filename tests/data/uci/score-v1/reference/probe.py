# SPDX-License-Identifier: GPL-3.0-or-later
"""Original CPU-only official-binary score protocol probe; research only."""
from __future__ import annotations
import hashlib
import json
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXE=Path('<workspace>/.toolchain/reference/lc0/lc0.exe')
EXE_SHA='e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b'
FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def run_session(name: str, backend: str, backend_opts: str, score_type: str|None=None, fen: str=FEN, *, go: str='go nodes 1', show_wdl: bool=True):
    if backend not in ('trivial','random','recordreplay'):
        raise ValueError('CPU-only backend allowlist')
    if backend=='recordreplay' and not backend_opts.startswith('backend=random,random('):
        raise ValueError('Record/replay must have explicit CPU-only parent and child')
    if hashlib.sha256(EXE.read_bytes()).hexdigest()!=EXE_SHA:
        raise ValueError('Official binary hash changed')
    directory=ROOT/name
    directory.mkdir(exist_ok=False)
    argv=[str(EXE),'--config=',f'--backend={backend}',f'--backend-opts={backend_opts}',
          '--weights=','--threads=1','--minibatch-size=1','--max-prefetch=0',
          '--nncache=0','--task-workers=0','--verbose-move-stats']
    if show_wdl: argv.append('--show-wdl')
    if score_type is not None: argv.append('--score-type='+score_type)
    (directory/'argv.json').write_text(json.dumps(argv,indent=2)+'\n',encoding='utf-8')
    stdout=[]; stderr=[]; commands=[]; lines=queue.Queue()
    process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',bufsize=1)
    def reader(stream, retained, publish=False):
        try:
            for line in stream:
                retained.append(line)
                if publish: lines.put(line)
        finally:
            if publish: lines.put(None)
    workers=[threading.Thread(target=reader,args=(process.stdout,stdout,True)),
             threading.Thread(target=reader,args=(process.stderr,stderr))]
    for worker in workers: worker.start()
    def send(line):
        commands.append(line)
        process.stdin.write(line+'\n'); process.stdin.flush()
    def until(prefix, seconds=15):
        deadline=time.monotonic()+seconds
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError(prefix)
            line=lines.get(timeout=remaining)
            if line is None: raise RuntimeError('Early Lc0 EOF: '+''.join(stderr))
            if line.startswith(prefix): return line.rstrip('\r\n')
    try:
        send('uci'); until('uciok')
        send('isready'); until('readyok')
        send('ucinewgame')
        send('position fen '+fen)
        send(go)
        best=until('bestmove ')
        send('quit'); process.stdin.close(); process.wait(timeout=10)
        if process.returncode!=0: raise RuntimeError('Lc0 exit '+str(process.returncode))
        return {'bestmove':best,'info':[x.rstrip() for x in stdout if x.startswith('info ') and ' score ' in x]}
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
        for worker in workers: worker.join(timeout=5)
        (directory/'commands.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')
        (directory/'stdout.txt').write_text(''.join(stdout),encoding='utf-8')
        (directory/'stderr.txt').write_text(''.join(stderr),encoding='utf-8')
        (directory/'exit.json').write_text(json.dumps({'exit':process.returncode})+'\n',encoding='utf-8')

def records(path):
    data=path.read_bytes(); offset=0; rows=[]
    while offset<len(data):
        key,count=struct.unpack_from('<Qi',data,offset);offset+=12
        if count<0 or count>10000 or offset+count*4>len(data): raise ValueError('Invalid record')
        values=list(struct.unpack_from('<'+'f'*count,data,offset));offset+=count*4
        rows.append((key,values))
    return rows

if __name__=='__main__':
    record=ROOT/'record-root.bin'
    options="backend=random,random(seed=0),record_file='"+record.as_posix()+"'"
    print(json.dumps(run_session('record-root','recordreplay',options),indent=2))
    print('records',records(record))
