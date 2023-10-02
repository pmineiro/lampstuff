from CS import EmpBernCS

class GammaAutoTune(object):
    def __init__(self, model, sampler):
        
        super().__init__()
        self.sampler = sampler
        self.gammas = [1, 2]
        self.models = [ model.clone(), model.clone() ]
        self.alpha = 0.05
        self.nswitches = 0
        self.cs = EmpBernCS(self.alpha / 2)

    def sample(self, x):
        fhats = [ m.predict(x) for m in self.models ]
        return [ self.sampler(fhat, gamma) for fhat, gamma in zip(fhats, self.gammas) ]

    def bandit_learn(self, x, a, r):
        loss = sum(m.bandit_learn(x, ma, mr) 
                   for m, apair, rpair in zip(self.models, a, r)
                   for ma, mr in zip(apair, rpair)
                   )
        self.__csupdate([ (rbig - rsmall).item() for rbig, rsmall in zip(r[1][0], r[0][0]) ])

        return loss

    def masked_bandit_learn(self, x, samples, rewards):
        greedy = []
        loss = 0
        for m, (explore, exploit), r in zip(self.models, samples, rewards):
            o = explore.clone()
            o[range(o.shape[0]), exploit] = 1
            loss += m.masked_bandit_learn(x, o, r)
            greedy.append(r[range(r.shape[0]), exploit])

        self.__csupdate([ (rbig - rsmall).item() for rbig, rsmall in zip(greedy[1], greedy[0]) ])

        return loss

    def __csupdate(self, dvs):
        for dv in dvs: self.cs.addobs(dv)
        if self.cs.is_at_least(0) or self.cs.is_at_most(0):
            if self.cs.is_at_least(0):
                self.gammas = [ self.gammas[1], 2 * self.gammas[1] ]
                self.models = [ self.models[1], self.models[1].clone() ]
            else:
                self.gammas = [ 1/2 * self.gammas[0], self.gammas[0] ]
                self.models = [ self.models[0].clone(), self.models[0] ]

            # NB: since the test is bidirectional, we control FPR instead of type 1
            #     i.e., no alpha spending here
            self.nswitches += 1
            self.cs = EmpBernCS(self.alpha / 2)
