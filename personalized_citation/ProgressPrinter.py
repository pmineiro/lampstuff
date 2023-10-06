class ProgressPrinter(object):
    def __init__(self, *header):
        super().__init__()
        self.rawheader = header
        self.width = max(14, max(len(h) + 8 for h in self.rawheader))
        self.autoprint = True
        self.extra = None

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, val):
        self._extra = val
        return self._extra

    @property
    def autoprint(self):
        return self._autoprint

    @autoprint.setter
    def autoprint(self, val):
        self._autoprint = val
        return self._autoprint

    def addobs(self, *observation):
        for n, v in enumerate(observation):
            if v is not None:
                self.n[n] += 1
                self.sum[n] += v
                self.nsincelast[n] += 1
                self.sincelast[n] += v

        self.cnt += 1
        if self.autoprint and self.cnt and (self.cnt & (self.cnt - 1)) == 0:
            self.print()

    def format_time(self, dt):
        if dt < 1:
            return f'{1000*dt:>4.3g} ms'
        elif dt < 60:
            return f'{dt:>5.3g} s'
        elif dt < 60 * 60:
            return f'{dt/60:>5.3g} m'
        elif dt < 24 * 60 * 60:
            return f'{dt/(60*60):>5.3g} h'
        elif dt < 7 * 24 * 60 * 60:
            return f'{dt/(24*60*60):>5.3g} d'
        else:
            return f'{dt/(7*24*60*60):>5.3g} w'

    def print(self):
        if any(self.nsincelast):
            import time

            end = time.time()

            print(' '.join([ f'{self.cnt:<7d}' ] +
                           [ f'{v[0]:{self.width-8}.3g} ({v[1]:5.3g})'
                             for n, s in enumerate(self.rawheader)
                             for v in ((self.sum[n]/max(1,self.n[n]), self.sincelast[n]/max(1,self.nsincelast[n]),),)
                           ] +
                           [ self.format_time(end - self.start) ]),
                  flush=True)
            self.nsincelast = [0] * len(self.rawheader)
            self.sincelast = [0] * len(self.sincelast)

            if callable(self.extra):
                self.extra()

    def __enter__(self):
        import time

        self.fullheader = ['n'] + [ f'{what} (since)' for what in self.rawheader ] + ['dt']
        print(' '.join([ f'{h:<7s}' if n == 0 else f'{h:>7s}' if h == 'dt' else f'{h:>{self.width}s}' for n, h in enumerate(self.fullheader) ]), flush=True)
        self.cnt = 0
        self.n = [0] * len(self.rawheader)
        self.sum = [0] * len(self.rawheader)
        self.nsincelast = [0] * len(self.rawheader)
        self.sincelast = [0] * len(self.rawheader)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.print()
