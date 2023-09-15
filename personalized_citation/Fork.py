import os
import sys
import traceback

class SubProcess(object):
    def __init__(self):
        super().__init__()

    @property
    def parent(self):
        return self.pid > 0

    def __enter__(self):
        self.pid = os.fork()
        if self.pid > 0:
            os.waitpid(self.pid, 0)

        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.pid == 0:
            if exc_type is not None:
                print(exc_type, file=sys.stderr, flush=True)
                print(exc_value, file=sys.stderr, flush=True)
                traceback.print_tb(tb, file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
