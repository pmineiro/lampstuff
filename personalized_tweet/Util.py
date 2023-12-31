def interleave(a, b, *, sequential=False):
    if sequential:
        yield from ( (True, v) for v in a )
        yield from ( (False, v) for v in b )
    else:
        from math import inf

        atot, btot = a.num_examples, b.num_examples
        aiter, biter = a.__iter__(), b.__iter__()
        aelem, belem = next(aiter), next(biter)
        anum, bnum = 1, 1

        while anum != inf or bnum != inf:
            if anum * btot <= bnum * atot:
                yield (True, aelem)
                try:
                    aelem = next(aiter)
                    anum += 1
                except StopIteration:
                    anum = inf
            else:
                yield (False, belem)
                try:
                    belem = next(biter)
                    bnum += 1
                except StopIteration:
                    bnum = inf

class Filter(object):
    def __init__(self, stream, re_pattern):
        import re

        self.stream = stream
        self.pattern = re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if len(data) and self.pattern.search(data) is None:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()

import contextlib
@contextlib.contextmanager
def BadPipe():
    import sys
    save = (sys.stderr, sys.__stderr__)
    sys.stderr = Filter(sys.stderr, r'Bad pipe message|\[b|^\s+$') if sys.stderr else None
    sys.__stderr__ = Filter(sys.__stderr__, r'Bad pipe message|\[b|^\s+$') if sys.__stderr__ else None
    yield
    sys.stderr, sys.__stderr__ = save

from pathlib import Path
import os

@contextlib.contextmanager
def set_directory(path: Path):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

class ShufBuf(object):
    def __init__(self, gen, *, bufsize, seed):
        super().__init__()
        self.gen = gen
        self.bufsize = bufsize
        self.seed = seed

    def __iter__(self):
        def items():
            import random

            buf = [None]*self.bufsize
            rand = random.Random(self.seed)
            for v in self.gen:
                index = rand.randrange(self.bufsize)
                if isinstance(buf[index], tuple):
                    yield buf[index][0]
                buf[index] = (v,)

            yield from (v[0] for v in buf if isinstance(v, tuple))

        return items()
