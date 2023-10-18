#! /usr/bin/env python

import sys
sys.path.insert(0, '.')

import argparse
from contextlib import redirect_stdout
from json import load, dump
from pathlib import Path
from Util import set_directory

parser = argparse.ArgumentParser(description='Run a notebook cell at the terminal.')
parser.add_argument('filename', help='notebook to run')
parser.add_argument('cellnum', metavar='n', type=int, help='cell number to run')
parser.add_argument('out', help='file to output updated notebook')
parser.add_argument('-y', '--yes', action='store_true', help='yes to all questions')
args = parser.parse_args()

class Logger(object):
    def __init__(self):
        import sys

        self.terminal = sys.stdout
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)  

    def flush(self):
        self.terminal.flush()

    def nicelog(self):
        for a, b in zip(self.log, self.log[1:]):
            if b == '\n':
                yield a + '\n'
            elif a != '\n':
                yield a

        if self.log and self.log[-1] != '\n':
            yield self.log[-1]

def user_confirm(question: str) -> bool:
    if args.yes:
        return True

    reply = str(input(question + ' (y/N): ')).lower().strip()
    if reply[:1] == 'y':
        return True
    else:
        return False

with open(args.filename) as fp:
    nb = load(fp)

num = 0
for rawnum, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        num += 1
        if num == args.cellnum + 1:
            source = ''.join(line for line in cell['source'] if not line.startswith('%'))
            sourcelines = source.split('\n')
            first = '\n'.join(sourcelines[:5])
            last = '\n'.join(sourcelines[-5:])
            if user_confirm(f'execute {first}\n...\n{last}\n'):
                try:
                    path = Path(args.filename)
                    log = Logger()
                    with redirect_stdout(log), set_directory(path.parent):
                        exec(source, {}, {})
                finally:
                    outputs = cell.get('outputs', [])
                    outputs.append({ 'name': 'stdout', 'output_type': 'stream', 'text': list(log.nicelog()) })
                    nb['cells'][rawnum]['outputs'] = outputs
                    
                    from pprint import pformat
                    print(pformat(outputs))
                    print(pformat(log.log))
                    with open(args.out, 'w') as fp:
                        dump(nb, fp, indent=2)

# [ {'name': 'stdout', 'output_type': 'stream', 'text': ["",""...] } ]
