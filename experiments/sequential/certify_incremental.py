"""
Bound every fixed Bernoulli bet using exact rational arithmetic.

For p=7/10, m=1/2, delta=9/10 and milestones 2,4,8, partition
b=lambda/2 in [0,1). On each interval, pay a milestone at the first node
(n,k) where *some* b in that interval could reach it. These optimistic
crossings dominate those of every fixed bet in the interval, path by path.
Integer first-passage recursion and a rational tail bound then give a global
upper bound, including isolated milestone equalities and the open endpoint.
No root finding, state interpolation, or floating-point crossing tests occur.

With --adaptive, also validate two fixed numerical adaptive policies on fresh
simulations, with downward-rounded evidence and payments, and show that their
simultaneous 99% Hoeffding lower bounds exceed the all-fixed upper bounds.
This certifies the fixed-policy upper bounds deterministically and the
adaptive separation with the stated statistical confidence; it does not
certify adaptive optimality.

Writes results/certify_incremental.json and, with --adaptive,
results/incremental_certificate_table.tex. The exact arithmetic needs only
Python's standard library.
"""

import argparse
from fractions import Fraction as Q
import json
import math

from common import save_text


LEVELS = (2, 4, 8)
PAYOUTS = {"early": (Q(8, 5), Q(2, 5), Q(4, 5)),
           "late": (Q(1, 5), Q(2, 5), Q(32, 5))}


def possible_crossings(n, k, left, right):
    """Return possible crossings in [left,right], excluding b=1.

    (1+b)^k(1-b)^(n-k) is maximized at b=(2k-n)/n, clipped
    to the interval. At b=1 only all-success nodes can have a positive
    supremum; equality there is not a crossing in the admissible domain.
    """
    peak = min(right, max(left, Q(2*k-n, n)))
    num, den = peak.numerator, peak.denominator
    numerator = (den+num)**k * (den-num)**(n-k)
    denominator = den**n
    if peak == 1:
        return tuple(numerator > a*denominator for a in LEVELS)
    return tuple(numerator >= a*denominator for a in LEVELS)


def interval_values(left, right, horizon):
    # The discount-weighted transition probabilities are 63/100 and 27/100.
    # Store all live and absorbed mass over the common denominator 100**n.
    live = [[1] for _ in LEVELS]
    hit = [0]*len(LEVELS)
    for n in range(1, horizon+1):
        crossed = [possible_crossings(n, k, left, right) for k in range(n+1)]
        for j in range(len(LEVELS)):
            nxt = [0]*(n+1)
            for k, mass in enumerate(live[j]):
                nxt[k] += 27*mass
                nxt[k+1] += 63*mass
            hit[j] *= 100
            for k in range(n+1):
                if crossed[k][j]:
                    hit[j] += nxt[k]
                    nxt[k] = 0
            live[j] = nxt
    return [Q(x, 100**horizon) for x in hit]


def upward_decimal(value, places=6):
    scale = 10**places
    rounded = -(-value.numerator*scale // value.denominator)
    return f"{rounded//scale}.{rounded%scale:0{places}d}"


def certify(partitions=256, horizon=50):
    best = {name: Q(0) for name in PAYOUTS}
    locations = {}
    for i in range(partitions):
        hits = interval_values(Q(i, partitions), Q(i+1, partitions), horizon)
        for name, payouts in PAYOUTS.items():
            value = sum(r*h for r, h in zip(payouts, hits))
            if value > best[name]:
                best[name], locations[name] = value, i
    result = {"p": "7/10", "m": "1/2", "delta": "9/10",
              "horizon": horizon, "partitions": partitions,
              "method": "exact rational interval crossing upper bound",
              "allocations": {}}
    for name, payouts in PAYOUTS.items():
        tail = Q(9, 10)**(horizon+1)*sum(payouts)
        upper = best[name]+tail
        result["allocations"][name] = {
            "finite_upper": str(best[name]), "tail_upper": str(tail),
            "infinite_upper": str(upper),
            "infinite_upper_decimal_rounded_up": upward_decimal(upper),
            "maximizing_interval_index": locations[name]}
    return result


def check():
    # Excluded endpoint: three successes approach 8 but cannot reach it.
    assert possible_crossings(3, 3, Q(99, 100), Q(1)) == (True, True, False)
    # An included exact hit must be counted; b=1/2 gives E_2=9/4.
    assert possible_crossings(2, 2, Q(1, 2), Q(1, 2))[0]
    assert interval_values(Q(0), Q(0), 4) == [0, 0, 0]
    # With b=1/2, the first possible hit of level 2 is two successes.
    assert interval_values(Q(1, 2), Q(1, 2), 2) == [Q(63, 100)**2, 0, 0]
    # Subdividing intervals can only improve the optimistic bound.
    for name in PAYOUTS:
        coarse = certify(2, 6)["allocations"][name]["finite_upper"]
        fine = certify(4, 6)["allocations"][name]["finite_upper"]
        assert Q(fine) <= Q(coarse)
    # Check downward evidence propagation against the exact rational product,
    # including nearly maximal bets and many successive roundings.
    evidence, lower = Q(1), 1.
    for t in range(100):
        bet = (0., .8, 1.363585661028494, 1.99)[t % 4]
        sign = .5 if t % 3 else -.5
        evidence *= 1+Q(bet)*Q(sign)
        factor = math.nextafter(1+bet*sign, -math.inf)
        lower = math.nextafter(lower*factor, -math.inf)
        assert Q(lower) <= evidence


def adaptive_lower_bounds(n_sim=500000, alpha=.01):
    """Fresh validation of two fixed numerical policies, jointly at 1-alpha.

    Track a downward-rounded evidence bound and use it to select actions,
    update the internal milestone stage, and make termination decisions.
    This defines a predictable policy of its own: the Bellman construction
    matches incremental_milestones.py, but the policy is solved and evaluated
    separately and uses a rounded state instead of floating-point log evidence.
    Round each multiplier and evidence product toward -infinity with nextafter.
    Under these same chosen actions, true evidence dominates the tracked bound.
    Count payments only when the bound crosses; delayed or omitted payments
    lower discounted receipts. Discount factors, payments, and their sums are
    rounded downward too. Each simulated observation is therefore bounded by
    the actual policy payoff. Hoeffding and a union bound cover both allocations.
    """
    import numpy as np
    from statcontracts.sequential import BettingReward
    from statcontracts.sequential.milestones import (
        policy_bets, solve_milestone_bellman)

    p, m, delta = .7, .5, .9
    reward = BettingReward(p, m)
    _, cgf = reward.interior_cgf_optimal(delta)
    results = {}
    for index, (name, rational_payouts) in enumerate(PAYOUTS.items()):
        payouts = np.array([float(x) for x in rational_payouts])
        control = solve_milestone_bellman(
            reward, delta, LEVELS, payouts, 4800, 401, -12, 1.99,
            extra_actions=(.8, cgf, 2*(8**.25-1)*(1+1e-11)))
        horizon = 200
        rng = np.random.default_rng(20261017+index)
        evidence_lower = np.ones(n_sim)
        stage = np.zeros(n_sim, dtype=int)
        receipts = np.zeros(n_sim)
        # Downward-round every term of the discounted payout and its sum.
        discount_lower = 1.
        delta_lower = np.nextafter(delta, -np.inf)
        payouts_lower = np.nextafter(payouts, -np.inf)
        for t in range(1, horizon+1):
            discount_lower = np.nextafter(discount_lower*delta_lower, -np.inf)
            active = np.flatnonzero(stage < len(LEVELS))
            if not len(active):
                break
            bets = policy_bets(control, stage[active],
                               np.log(evidence_lower[active]))
            observations = rng.random(len(active)) < p
            # bet*(X-1/2) is exact binary scaling; nextafter encloses rounding
            # of the addition and multiplication from below.
            factor_lower = np.nextafter(1+bets*(observations-m), -np.inf)
            evidence_lower[active] = np.nextafter(
                evidence_lower[active]*factor_lower, -np.inf)
            for k, level in enumerate(LEVELS):
                crossed = (stage[active] <= k) & (evidence_lower[active] >= level)
                paid = active[crossed]
                amount = np.nextafter(discount_lower*payouts_lower[k], -np.inf)
                receipts[paid] = np.nextafter(receipts[paid]+amount, -np.inf)
            stage[active] = np.maximum(stage[active], np.searchsorted(
                LEVELS, evidence_lower[active], side="right"))
            stage[active[np.log(evidence_lower[active]) < control["lower"]]] = len(LEVELS)
        # Each receipt is in [0, delta*sum r]. This analytical bound includes
        # every future payment, although the sampled lower payoff stops at 200.
        bound = float(Q(9, 10)*sum(rational_payouts))
        radius = bound*math.sqrt(math.log(2/alpha)/(2*n_sim))
        mean = math.fsum(receipts)/n_sim
        results[name] = {"mean_lower_receipts": mean,
            "hoeffding_radius": radius, "value_lower": mean-radius-1e-12,
            "n_sim": n_sim, "horizon": horizon, "seed": 20261017+index,
            "joint_confidence": 1-alpha, "evidence_rounding": "downward",
            "policy_grid_value": control["value"]}
        print(name, "adaptive lower:", results[name], flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--adaptive", action="store_true",
                        help="also validate adaptive policies on fresh simulations")
    args = parser.parse_args()
    check()
    result = certify(args.partitions, args.horizon)
    if args.adaptive:
        result["adaptive_validation"] = adaptive_lower_bounds()
        lines = [r"% Generated by experiments/sequential/certify_incremental.py --adaptive.",
                 r"\begin{tabular}{lrr}", r"\hline",
                 r"Allocation & Every fixed parameter: upper bound & Adaptive policy: lower bound \\",
                 r"\hline"]
        for name, row in result["allocations"].items():
            lower = result["adaptive_validation"][name]["value_lower"]
            assert lower > float(Q(row["infinite_upper"]))
            lower_display = math.floor(1e6*lower)/1e6
            lines.append(f"{name.capitalize()} & {row['infinite_upper_decimal_rounded_up']} & {lower_display:.6f}"+r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        save_text("incremental_certificate_table.tex", "\n".join(lines)+"\n")
    save_text("certify_incremental.json", json.dumps(result, indent=2)+"\n")
    for name, row in result["allocations"].items():
        print(name, "all-fixed upper bound:", row["infinite_upper_decimal_rounded_up"])
