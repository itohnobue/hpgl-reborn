from math import sqrt, log
from copy import copy
import numpy as np


# Computes zk such that P(Z<z) = p.
# This function uses a numerical approximation from : Statistical Computing,
# by W.J. Kennedy, Jr. and James E. Gentle, 1980, p. 95.

def inverse_normal_score(prob, mean=0.0, var=1.0):
    Pconst = [-0.322232431088, -1.0, -0.342242088547, -0.0204231210245, -0.0000453642210148]
    Qconst = [0.0993484626060, 0.588581570495, 0.531103462366, 0.103537752850, 0.0038560700634]

    if prob >= 1.0:
        return 3
    elif prob <= 0.0:
        return -3

    tmp_prob = prob
    if prob > 0.5:
        tmp_prob = 1 - prob

    y = sqrt(log(1.0 / (tmp_prob * tmp_prob)))
    num = (((y * Pconst[4] + Pconst[3]) * y + Pconst[2]) * y + Pconst[1]) * y + Pconst[0]
    denom = (((y * Qconst[4] + Qconst[3]) * y + Qconst[2]) * y + Qconst[1]) * y + Qconst[0]

    result = y + num / denom

    if prob == tmp_prob:
        result = -result

    R = result * sqrt(var) + mean
    return R


def normal_score(prob):
    a1 = 0.4361836
    a2 = -0.1201676
    a3 = 0.9372980
    p = 0.33267
    # F-45: the A&S 26.2.17 approximation is only valid for x >= 0; for
    # negative x it returns garbage (e.g. x=-3 -> -526978). Use the CDF
    # symmetry P(x) = 1 - P(-x) for negative inputs.
    if prob < 0:
        return 1.0 - normal_score(-prob)
    z = np.exp(-(prob ** 2) / 2) / (sqrt(2 * np.pi))
    t = 1.0 / (1 + p * prob)
    P_x = 1 - z * (a1 * t + a2 * (t ** 2) + a3 * (t ** 3))
    return P_x


def gaussian_cdf(value, mean=0.0, var=1.0):
    p = 0.2316419
    b = np.array([0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429])
    sx = (value - mean) / sqrt(var)

    neg = sx < 0

    # F-07: the A&S 26.2.17 series is valid for x >= 0. Negate NEGATIVE
    # inputs (not positive ones) before the series, then apply the CDF
    # symmetry result = 1 - result for those negatives. Both branches were
    # inverted, corrupting the 7.3 back-transform (z=3 -> 1.565 vs 0.9987).
    if neg:
        sx = -sx

    t = 1 / (1 + p * sx)

    zx = 1 / sqrt(2 * 3.14159265358) * np.exp(-sx * sx / 2)
    result = 1 - zx * ((((b[4] * t + b[3]) * t + b[2]) * t + b[1]) * t + b[0]) * t

    if neg:
        result = 1 - result

    return result


def cdf_transform(array_data, undefined_value):
    array_copy = copy(array_data)
    defined_mask = array_copy != undefined_value
    defined_values = array_copy[defined_mask]
    defined_values_count = float(defined_values.size)
    if defined_values_count == 0:
        raise ValueError("cdf_transform: no defined values (all cells are undefined)")
    # II-01: assign quantiles by VALUE RANK, not grid position. The old
    # code accumulated a running proportion while walking the grid in
    # (i, j) order, so (a) tied values received different quantiles and
    # (b) the cumulative probability overflowed 1.0 (-> inverse_normal_score
    # clamps to +3.0); back_cdf_transform could not invert the table.
    # Sort the defined values, give each unique value the cumulative
    # probability of its rank (ties share one quantile), and transform
    # every defined cell to the quantile of its value.
    unique_values, counts = np.unique(defined_values, return_counts=True)
    cum_probs = np.cumsum(counts) / defined_values_count
    # R-08 (II-01): the final cumulative probability is exactly 1.0 (sum of
    # counts == defined_values_count). A p=1.0 rank maps to the inverse
    # normal score clamp +3.0, whose gaussian_cdf is only 0.99865 — the
    # back-transform then returns the 99.865th-percentile value, not the
    # max (empirical: 7.5-27.5% error at the max datum). Clamp the tail to
    # the largest float64 strictly below 1.0 (nextafter) so the max datum
    # maps to a large-but-finite normal score and round-trips exactly
    # (mirrors the library F-04 clamp at src/geo_bsd/cdf.py:218-226).
    cum_probs[-1] = np.nextafter(1.0, 0.0)
    for i in range(array_copy.shape[0]):
        for j in range(array_copy.shape[1]):
            value = array_copy[i, j]
            if value != undefined_value:
                rank = np.searchsorted(unique_values, value)
                # .item() unwraps a 1-element array (for (n, m, 1) inputs,
                # value/rank are shape-(1,) arrays) back to a scalar —
                # inverse_normal_score uses math.sqrt/log (scalar-only).
                prob = np.asarray(cum_probs[rank]).item()
                array_data[i, j] = inverse_normal_score(prob)
    return cum_probs, unique_values


def back_cdf_transform(property_arr, props, values, undefined_value):
    for i in range(property_arr.shape[0]):
        for j in range(property_arr.shape[1]):
            for k in range(property_arr.shape[2]):
                if property_arr[i, j, k] != undefined_value:
                    property_arr[i, j, k] = np.interp(gaussian_cdf(property_arr[i, j, k]), props, values)
