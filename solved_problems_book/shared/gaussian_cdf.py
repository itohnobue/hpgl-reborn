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
    z = np.exp(-(prob ** 2) / 2) / (sqrt(2 * np.pi))
    t = 1.0 / (1 + p * prob)
    P_x = 1 - z * (a1 * t + a2 * (t ** 2) + a3 * (t ** 3))
    return P_x


def gaussian_cdf(value, mean=0.0, var=1.0):
    p = 0.2316419
    b = np.array([0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429])
    sx = (value - mean) / sqrt(var)

    neg = sx < 0

    if not neg:
        sx = -sx

    t = 1 / (1 + p * sx)

    zx = 1 / sqrt(2 * 3.14159265358) * np.exp(-sx * sx / 2)
    result = 1 - zx * ((((b[4] * t + b[3]) * t + b[2]) * t + b[1]) * t + b[0]) * t

    if not neg:
        result = 1 - result

    return result


def cdf_transform(array_data, undefined_value):
    array_copy = copy(array_data)
    value = 0.0
    props = np.array([])
    values = np.array([])
    defined_values_count = float(np.sum(array_copy != undefined_value))
    for i in range(array_copy.shape[0]):
        for j in range(array_copy.shape[1]):
            if array_copy[i, j] != undefined_value:
                value += float(np.sum(array_copy == array_copy[i, j])) / defined_values_count
                props = np.append(props, value)
                values = np.append(values, array_copy[i, j])
                array_data[i, j] = inverse_normal_score(value)
    return props, values


def back_cdf_transform(property_arr, props, values, undefined_value):
    for i in range(property_arr.shape[0]):
        for j in range(property_arr.shape[1]):
            for k in range(property_arr.shape[2]):
                if property_arr[i, j, k] != undefined_value:
                    property_arr[i, j, k] = np.interp(gaussian_cdf(property_arr[i, j, k]), props, values)
