"""
Check fixed-parameter thresholds for all three signs of the drift.

Nearest-neighbor walks have an independent closed form for the maximum at a
geometric horizon. Compare the Spitzer calculation (beta_threshold) with it
and check finite caps against finite-horizon optimal stopping, including
initial scales above the stopping boundary where the compulsory first trial
changes the value. Uses neither Monte Carlo nor state interpolation.

Run: python check_exact_threshold.py
"""

import numpy as np

from statcontracts.sequential import BettingReward, beta_threshold, sim_threshold


def main():
    # These betting factors are exp(+1) and exp(-1).
    m = 1/(np.e+1)
    lam = np.e-np.exp(-1)
    for p in (.25, .5, .75):
        for delta in (.8, .95):
            z = (1+np.sqrt(1-4*delta**2*p*(1-p)))/(2*delta*p)
            beta = 1/(z-1)
            actual = beta_threshold(BettingReward(p, m), lam, delta)
            assert abs(actual-beta) < 2e-11, (p,delta,actual,beta)
            def continuation_value(y):
                # M at a geometric horizon has P(M >= k) = z**(-k).
                # Hence E[(y + M - beta)^+] has this closed form.
                k = max(0, int(np.ceil(beta-y)))
                return (y+k)*z**(-k)

            for x in (-1., 0., beta, beta+.5, beta+2):
                limiting = delta*(p*continuation_value(x+1)
                                  + (1-p)*continuation_value(x-1))
                if x <= beta:
                    assert abs(limiting-continuation_value(x)) < 2e-11
                horizon = 800
                live = np.array([1.])
                hit_value = 0.
                for n in range(1,horizon+1):
                    nxt = np.zeros(n+1)
                    nxt[:-1] += (1-p)*live
                    nxt[1:] += p*live
                    w = x+2*np.arange(n+1)-n
                    hit = w >= beta
                    hit_value += delta**n*(nxt[hit]@w[hit])
                    nxt[hit] = 0
                    live = nxt
                capped = hit_value+delta**horizon*(live@w)
                v = x+np.arange(-horizon,horizon+1,dtype=float)
                for n in reversed(range(1,horizon)):
                    w = x+np.arange(-n,n+1)
                    v = np.maximum(w,delta*(p*v[2:]+(1-p)*v[:-2]))
                optimal = delta*(p*v[2]+(1-p)*v[0])
                assert capped <= optimal+1e-12 <= limiting+2e-11
                assert abs(capped-limiting) < 2e-11
            print(f"p={p}, delta={delta}: beta={beta:.8f}; "
                  "five initial scales checked")
    assert beta_threshold(BettingReward(.7, .5), 0, .9) == 0
    # With constant evidence and C > 1, the first possible collection is
    # delta*log(C), even though the reward threshold is already below C.
    payoffs, truncated = sim_threshold(BettingReward(.7, .5), 0, .9, np.exp(2),
                                       0, n_sim=10, max_T=5)
    assert np.allclose(payoffs, 1.8) and truncated == 0
    print("Positive-, zero-, and negative-drift threshold checks passed.")


if __name__ == "__main__":
    main()
