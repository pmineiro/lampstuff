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
            if self.pattern.search(data) is None:
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
    save = sys.__stderr__
    sys.__stderr__ = Filter(sys.__stderr__, r'Bad pipe message|\[b')
    yield
    sys.__stderr__ = save
