class ReplayWrapper(object):
    def __init__(self, model, replay_count, nbatches):
        from pyskiplist import SkipList

        assert replay_count == int(replay_count) and replay_count >= 0
        assert replay_count <= nbatches
        
        super().__init__()
        self.model = model
        self.replay_count = replay_count
        self.nbatches = nbatches
        self.replay_buffer = SkipList()
        self.n = 0

    def clone(self):
        from copy import deepcopy

        other = ReplayWrapper(self.model.clone(), self.replay_count, self.nbatches)
        other.replay_buffer = deepcopy(self.replay_buffer)

        return other

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def __replay(self):
        import torch

        perm = torch.randperm(len(self.replay_buffer), device='cpu')
        delme = []
        for idx in perm[:self.replay_count].tolist():
            n, (method, args, kwargs, cntptr) = self.replay_buffer[idx]
            dafunc = getattr(self.model, method)
            dafunc(*args, **kwargs)
            cntptr[0] -= 1
            if cntptr[0] <= 0:
                delme.append(n)
            
        for n in delme:
            self.replay_buffer.remove(n)

    def __maybe_replay(self, method, *args, **kwargs):
        if self.replay_count > 0:
            if len(self.replay_buffer) > self.nbatches:
                self.__replay()

            self.n += 1
            self.replay_buffer.insert( self.n, ( method, args, kwargs, [self.replay_count] ) )

        dafunc = getattr(self.model, method)
        return dafunc(*args, **kwargs)

    def learn(self, *args, **kwargs):
        return self.__maybe_replay('learn', *args, **kwargs)

    def bandit_learn(self, *args, **kwargs):
        return self.__maybe_replay('bandit_learn', *args, **kwargs)

    def masked_bandit_learn(self, *args, **kwargs):
        return self.__maybe_replay('masked_bandit_learn', *args, **kwargs)
