# The function of implied volatility
import numpy as np

def IV_b2_slope(tau, T_conv = 0.25):
    return np.exp(-1 * np.sqrt(tau/T_conv))


def IV_b3_moneyness_slope(M):
    M_pos = 1 if (M >= 0) else 0
    M_neg = 1 if (M < 0) else 0
    return M * M_pos + M_neg * (np.exp(2 * M) - 1) /(np.exp(2 * M) + 1)


def IV_b4_smile(tau ,M, T_max = 5):
    return (1 - np.exp(-1 * M**2)) * np.log(tau / T_max)


def IV_b5_smirk(tau ,M, T_max = 5):
    if M >= 0:
        a = 0
    else:
        a = (1 - np.exp((3 * M)**3)) * np.log(tau / T_max)
    return a

def IV_Remi(M, tau, beta):
    """
    :param beta: In R^5. Should be an 5-dim array.
    :param M: Log-moneyness
    :param tau: Annualized time-to-maturity
    :param T_max: Maximum maturity the model can achieve
    :param T_conv: Location of a fast convexity change
    :return: in R. THe implied volatility
    """

    b1 = beta.iloc[0]
    b2 = beta.iloc[1]
    b3 = beta.iloc[2]
    b4 = beta.iloc[3]
    b5 = beta.iloc[4]

    implied_vol = b1 + \
                  b2 * IV_b2_slope(tau=tau) +\
                  b3 * IV_b3_moneyness_slope(M) +\
                  b4 * IV_b4_smile(tau = tau, M = M) +\
                  b5 * IV_b5_smirk(tau = tau, M = M)

    return implied_vol
