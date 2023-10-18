#! /usr/bin/env python

import argparse
from contextlib import redirect_stdout
from json import load, dump

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

    def nicelog(self):
        for a, b in zip(self.log, self.log[1:]):
            if b == '\n':
                yield a + '\n'
            elif a != '\n':
                yield a

        if self.log[-1] != '\n':
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

for num, cell in enumerate([ v for v in nb['cells'] if v['cell_type'] == 'code']):
    if num == args.cellnum:
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        if user_confirm(f'execute {source[:256]} ...'):
            with redirect_stdout(Logger())as f:
                exec(source, globals(), locals())
            outputs = cell.get('outputs', [])
            outputs.append({ 'name': 'stdout', 'output_type': 'stream', 'text': list(f.nicelog()) })
            cell['outputs'] = outputs
            with open(args.out, 'w') as fp:
                dump(nb, fp)

# [ {'name': 'stdout', 'output_type': 'stream', 'text': ["",""...] } ]
