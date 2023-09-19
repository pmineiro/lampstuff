class ReplayWrapper(object):
    def __init__(self, model, replay_count, nbatches):
        from pyskiplist import SkipList

        assert replay_count == int(replay_count) and replay_count >= 0
        assert replay_count <= nbatches
        
        super().__init__()
        self.model = model
        self.replay_count = replay_count
        self.nbatches = nbatches
        self.learn_replay_buffer = SkipList()
        self.bandit_learn_replay_buffer = SkipList()
        self.n = 0

    def clone(self):
        from copy import deepcopy

        other = ReplayWrapper(self.model.clone(), self.replay_count, self.nbatches)
        other.learn_replay_buffer = deepcopy(self.learn_replay_buffer)
        other.bandit_learn_replay_buffer = deepcopy(self.bandit_learn_replay_buffer)

        return other

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def __learn_replay(self):
        import torch

        perm = torch.randperm(len(self.learn_replay_buffer), device='cpu')
        delme = []
        for idx in perm[:self.replay_count].tolist():
            n, (x, y, cntptr) = self.learn_replay_buffer[idx]
            self.model.learn(x, y)
            cntptr[0] -= 1
            if cntptr[0] <= 0:
                delme.append(n)
            
        for n in delme:
            self.learn_replay_buffer.remove(n)

    def learn(self, x, y):        
        if self.replay_count > 0:
            if len(self.learn_replay_buffer) > self.nbatches:
                self.__learn_replay()

            self.n += 1
            self.learn_replay_buffer.insert( self.n, ( x, y, [self.replay_count] ) )

        return self.model.learn(x, y)

    def __bandit_learn_replay(self):
        import torch

        perm = torch.randperm(len(self.bandit_learn_replay_buffer), device='cpu')
        delme = []
        for idx in perm[:self.replay_count].tolist():
            n, (x, a, r, cntptr) = self.bandit_learn_replay_buffer[idx]
            self.model.bandit_learn(x, a, r)
            cntptr[0] -= 1
            if cntptr[0] <= 0:
                delme.append(n)
            
        for n in delme:
            self.bandit_learn_replay_buffer.remove(n)

    def bandit_learn(self, x, a, r):   
        if self.replay_count > 0:
            if len(self.bandit_learn_replay_buffer) > self.nbatches:
                self.__bandit_learn_replay()

            self.n += 1
            self.bandit_learn_replay_buffer.insert( self.n, ( x, a, r, [self.replay_count] ) )

        return self.model.bandit_learn(x, a, r)
