# Bayesian Linear regression #
# The script gives the generalized OLS as spesified by Rémi(2022) Appendix A.2#

import SCRIPT.IV as IV
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import SCRIPT.Interpolator as itp

def regressors(options):
    """
    Build regressor of OLS_Sigma
    :param options: the options data at a given date
    :return: The regressors corresponding to Rémi 2022
    """

    y = options["impl_volatility"]
    x2 = IV.IV_b2_slope(options["tau"])
    x3 = options["log_moneyness"].apply(lambda x: IV.IV_b3_moneyness_slope(x))
    x4 = IV.IV_b4_smile(tau=options["tau"], M=options["log_moneyness"])
    x5 = options[["tau", "log_moneyness"]].apply(lambda x: IV.IV_b5_smirk(x.iloc[0], x.iloc[1]), axis=1)
    X = pd.concat([x2, x3, x4, x5], axis=1)
    X.columns = ["x2", "x3", "x4", "x5"]

    return X, y


def OLS(y, X):
    """
    :param y: The implied volatility
    :param X:  x2, x3, x4, x5 from IV expression
    :return: sigma of the OLS (the standard deviation)
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred_insample = model.predict(X)
    err = np.abs(y_pred_insample - y)
    sigma = np.sqrt(np.var(err))

    return sigma, model.coef_


def beta_prior(ATM_1y_t, ATM_1m_t, beat3_t_1, beta5_t_1):
    """
    :param ATM_1y_t: 1-year ATM IV
    :param ATM_1m_t: 1-month ATM IV
    :param beat3_t_1: calibrated beta3 at t-1
    :param beta5_t_1: calibrated beta5 at t-1
    :return: vector in R^4 (no prior for beta_4) as column vector
    """
    slope_t = (ATM_1y_t - ATM_1m_t)/np.exp(-1 * np.sqrt(4/12))
    beta_prior_t = np.array([ATM_1y_t,slope_t,beat3_t_1,beta5_t_1])

    return beta_prior_t.reshape([4,1])


def __priorVar__(options):

    # ATM option IV by interpolation
    IV_ATM_1year = options.groupby("date").apply(
        itp.Interpolate_IV
    )
    IV_ATM_1month = options.groupby("date").apply(lambda x:
                                                  itp.Interpolate_IV(x, maturity=20 / 252)
                                                  )
    IV_ATM_04 = options.groupby("date").apply(lambda x:
                                              itp.Interpolate_IV(x, log_moneyness=0.4, maturity=20 / 252)
                                              )

    slope = (IV_ATM_1year - IV_ATM_1month) / np.exp(-4 / 12)

    moneyness_slope_prox = IV_ATM_1month - IV_ATM_04

    # Prior var
    beta1_prior_var = np.var(IV_ATM_1year)
    beta2_prior_var = np.var(slope)
    beta3_prior_var = np.var(moneyness_slope_prox)
    beta5_prior_var = 1 * 1e-4

    beta_prior = [beta1_prior_var, beta2_prior_var, beta3_prior_var, beta5_prior_var]

    return beta_prior


def Bayesian_GLS_coef(prior_beta, sigma_t, X, y, prior_var):
    """
    Corresponding to the setting from Rémi(2022) page 112
    :param prior_var: The prior variance
    :return: GLS coefficient
    """
    # Infor matrix
    nrow = X.shape[0]
    R = np.delete(np.diag(np.repeat(1,5)), 3, axis =0)
    X = np.c_[np.repeat(1,nrow), X]
    XR = np.concatenate([X, R])

    # Dispersion Omega
    up_right = np.zeros([nrow,4])
    lower_left = np.zeros([4, nrow])

    up_left = np.eye(nrow) * sigma_t**2
    lower_right = np.diag(prior_var)



    Omega = np.block([[up_left, up_right],
                    [lower_left, lower_right]])


    # Y and Prior
    y = np.asarray(y).reshape([nrow, 1])
    yPrior = np.concatenate([y, prior_beta], axis=0)

    # GLS beta
    beta_post = np.linalg.inv(XR.T @ np.linalg.inv(Omega) @ XR) @ XR.T @ np.linalg.inv(Omega) @ yPrior

    return beta_post




