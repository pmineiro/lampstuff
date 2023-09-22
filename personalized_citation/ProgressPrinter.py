class ProgressPrinter(object):
    def __init__(self, *header):
        super().__init__()
        self.rawheader = header
        self.width = max(9, max(len(h) for h in self.rawheader))

    def addobs(self, *observation):
        for n, v in enumerate(observation):
            if v is not None:
                self.n[n] += 1
                self.sum[n] += v
                self.nsincelast[n] += 1
                self.sincelast[n] += v

        self.cnt += 1
        if self.cnt and (self.cnt & (self.cnt - 1)) == 0:
            self.print()

    def print(self):
        import time

        end = time.time()

        print(' '.join([ f'{self.cnt:<{self.width}d}' ] +
                       [ f'{v:{self.width}.3g}'
                         for n, s in enumerate(self.rawheader)
                         for v in (self.sum[n]/max(1,self.n[n]), self.sincelast[n]/max(1,self.nsincelast[n]),)
                       ] +
                       [ f'{v:{self.width}.3g}' for v in (end - self.start, ) ]),
              flush=True)
        self.nsincelast = [0] * len(self.rawheader)
        self.sincelast = [0] * len(self.sincelast)

    def __enter__(self):
        import time

        self.fullheader = ['n'] + [ h for what in self.rawheader for h in (what, 'since',) ] + ['dt (s)']
        print(' '.join([ f'{h:<{self.width}s}' if n == 0 else f'{h:>{self.width}s}' for n, h in enumerate(self.fullheader) ]), flush=True)
        self.cnt = 0
        self.n = [0] * len(self.rawheader)
        self.sum = [0] * len(self.rawheader)
        self.nsincelast = [0] * len(self.rawheader)
        self.sincelast = [0] * len(self.rawheader)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.print()
