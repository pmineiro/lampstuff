class EmpBernCS(object):
    def __init__(self, alpha):
        from math import log

        super().__init__()

        assert 0 < alpha <= 1

        self.sumv = 0
        self.sumvsq = 0
        self.n = 0
        self.alpha = alpha
        self.logalpha = log(self.alpha)

    def addobs(self, deltar):
        v = 1/2 + 1/2 * deltar
        assert 0 <= v <= 1
        self.sumv += v
        self.sumvsq += v*v
        self.n += 1

    def is_at_least(self, deltar):
        vtest = 1/2 + 1/2 * deltar

        logw = self.__logwealth(vtest, self.sumv, self.sumvsq, self.n)

        return logw >= -self.logalpha

    def is_at_most(self, deltar):
        vtest = 1/2 + 1/2 * deltar

        logw = self.__logwealth(vtest, self.n - self.sumv, self.n - 2 * self.sumv + self.sumvsq, self.n)

        return logw >= -self.logalpha

    def logwealth_at(self, deltar):
        vtest = 1/2 + 1/2 * deltar

        return self.__logwealth(vtest, self.sumv, self.sumvsq, self.n)

    @staticmethod
    # equivalent to log of Mathematica's \Gamma(k, 0, z)
    def logunnormalizedgamma(k, z):
        from math import log, exp
        from scipy.special import gammainc, loggamma

        try:
            rv = loggamma(k) + log(gammainc(k, z))
        except:
            # Series[Log[Gamma[k, 0, z] / Gamma[k, 0]], { z, Infinity, 1 }] // Normal
            # (-z + Log[Exp[z](-% /. Log[1 + x_] -> x)] // FullSimplify) /. Log[a_ / b_] -> Log[a] - Log[b] /. Gamma[a_, 0] -> Gamma[a]  /. Log[z^(-2 + k) a_] -> (k - 2) Log[z] + Log[a] // FullSimplify

            lgk = loggamma(k)
            logminusrv = -z + (k - 2) * log(z) + log(k + z - 1) - lgk # + O(1/z)
            rv = lgk - exp(logminusrv)

        return rv

    @staticmethod
    def __logwealth(mu, sumy, sumysq, t):
        from math import inf, log, sqrt

        assert 0 <= sumy
        assert 0 <= sumysq
        assert t >= 1

        rho = 16
        logrho = log(rho)
        loggammarhozerorho = EmpBernCS.logunnormalizedgamma(rho, rho)
        var = sumysq - sumy**2 / t                # optimal empirical variance
        var += 1 + 1 * log(t)                     # FTL regret bound for squared loss on [0, 1]
        loggammatvrhozerorho = EmpBernCS.logunnormalizedgamma(var, rho)

        tz = var + sumy - t * mu

        if rho + tz <= 1e-2:
            return -inf
        else:
            return rho * logrho - loggammarhozerorho + tz - (rho + var) * log(rho + tz) + EmpBernCS.logunnormalizedgamma(rho + var, rho + tz)

class EmpBernSuffixCS(EmpBernCS):
    def __init__(self, alpha):
        super().__init__()
        self.__restart()

    def __restart(self):
        from math import log
        self.sumv = 0
        self.sumvsq = 0
        self.n = 0
        self.logalpha = log(self.alpha / ( (self.nrestarts + 1) * (self.nrestarts + 2) ))
        self.nrestarts += 1

    def is_at_least(self, deltar):
        from math import log
        vtest = 1/2 + 1/2 * deltar

        logw = self.__logwealth(vtest, self.sumv, self.sumvsq, self.n)

        if logw < log(self.nrestarts + 1) - log(self.nrestarts + 3):
            self.__restart()
            return False
        else:
            return logw >= -self.logalpha
