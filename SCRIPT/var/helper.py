#############################################################
#################### Auxiliary ##############################
#############################################################

import numpy as np
import pandas as pd

from SCRIPT.Gibbs.gibbs_sampler import gibbs_sampler
from SCRIPT.var import var_cls
import matplotlib.pyplot as plt

def rolling_forecast(end,
                     exg,
                     lags,
                     window,
                     n,
                     burnin,
                     period = None,
                     pi=None,
                     save_plot = False,
                     own_lag_prior_mean = 1):
    '''
    :param end: endogenous variables
    :param exg: exogenous variables
    :param lags: number of leading lags
    :param window: rolling window width
    :param n: number of MCMC draws each step
    :param burnin: number of initial simulation to be dropped
    :param pi: prior variance size. list of 3 [pi1, pi2, pi3]
    :param period: number of steps to predict
    :param save_plot: True if save the plot of all values
    :param own_lag_prior_mean: gamma tilde own lag prior mean
    :return:
    '''

    Fhat_woex_B, Fhat_wex_B, Fhat_woex, Fhat_wex = [], [], [], []  # vault for projection

    if period is None:
        var = var_cls.var(end, exg, lags, pi, own_lag_prior_mean = own_lag_prior_mean)
        period = var.T - window - 1

    for i in range(window, window + period):
        print(i - window)
        cvx_var = var_cls.var(
            end.iloc[(i - window):i, ],
            exg,
            lags,
            pi=pi,
            own_lag_prior_mean = own_lag_prior_mean
        )

        # 1. Simulate the coefficient
        woex, _ = gibbs_sampler(cvx_var, n, ex=False)
        wex, _ = gibbs_sampler(cvx_var, n, ex=True)

        # 2. B-VAR forecast
        F_woex_B = []
        F_wex_B = []

        for j in range(burnin, n):
            _, yt1_woex = cvx_var.F(woex[j, :], h=1, ex=False)
            _, yt1_wex = cvx_var.F(wex[j, :], h=1, ex=True)
            F_woex_B = np.append(F_woex_B, yt1_woex)
            F_wex_B = np.append(F_wex_B, yt1_wex)

        F_woex_B = F_woex_B.reshape([n - burnin, cvx_var.m])
        F_wex_B = F_wex_B.reshape([n - burnin, cvx_var.m])

        # 3. VAR forecast
        F_woex, F_wex = cvx_var.var_h_prj()
        Fhat_woex = np.append(Fhat_woex, F_woex)
        Fhat_wex = np.append(Fhat_wex, F_wex)

        # 4. Compute the mean as estimation and store
        Fhat_woex_B = np.append(Fhat_woex_B, F_woex_B.mean(axis=0))
        Fhat_wex_B = np.append(Fhat_wex_B, F_wex_B.mean(axis=0))




    # Reshape the BVAR forecast
    Fhat_woex_B = pd.DataFrame(Fhat_woex_B.reshape([period, cvx_var.m]))
    Fhat_wex_B = pd.DataFrame(Fhat_wex_B.reshape([period, cvx_var.m]))
    Fhat_woex = pd.DataFrame(Fhat_woex.reshape([period, cvx_var.m]))
    Fhat_wex = pd.DataFrame(Fhat_wex.reshape([period, cvx_var.m]))


    # The realized beta
    test = end.iloc[window:(window+period), :].reset_index(drop=True)

    # Summarize MSE and plot
    MSE_woex_B = []
    MSE_wex_B = []
    MSE_woex = []
    MSE_wex = []
    for i in range(cvx_var.m):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
        ax.plot(test.iloc[:, i], label= r"Realized $\Delta \beta$")
        ax.plot(Fhat_woex_B.iloc[:, i], label="Without UMC (Bayesian)")
        ax.plot(Fhat_wex_B.iloc[:, i], label="With UMC (Bayesian)")
        ax.plot(Fhat_woex.iloc[:, i], label="Without UMC (VAR)")
        ax.plot(Fhat_wex.iloc[:, i], label="With UMC (VAR)")
        plt.legend()
        plt.title(r'$\pi_1=$' + str(cvx_var.pi1) + r'$, \pi_2=$' + str(cvx_var.pi2) + r', $\pi_3 = $' + str(cvx_var.pi3))
        plt.margins(x=0)
        if save_plot:
            plt.gcf()
            plt.savefig("OUTPUT/Plot/Hyper parameter check/" + str(cvx_var.pi1) + "_" + str(cvx_var.pi3) + "beta" +str(i+1)+ ".jpg")
        plt.show()

        mse_woex_B = sum(abs(Fhat_woex_B.iloc[:, i] - test.iloc[:, i])  / period)
        mse_wex_B = sum(abs(Fhat_wex_B.iloc[:, i] - test.iloc[:, i]) / period)
        mse_woex = sum(abs(Fhat_woex.iloc[:, i] - test.iloc[:, i]) / period)
        mse_wex = sum(abs(Fhat_wex.iloc[:, i] - test.iloc[:, i]) / period)

        MSE_woex_B = np.append(MSE_woex_B, mse_woex_B)
        MSE_wex_B = np.append(MSE_wex_B, mse_wex_B)
        MSE_woex = np.append(MSE_woex, mse_woex)
        MSE_wex = np.append(MSE_wex, mse_wex)

    MSE = pd.DataFrame([MSE_woex_B, MSE_wex_B, MSE_woex, MSE_wex],
                         index=["Without UMC (Bayesian VAR)", "With UMC (Bayesian VAR)",
                                "Without UMC (VAR)", "With UMC (VAR)"])

    # Set the correct index
    for i in [Fhat_woex_B, Fhat_wex_B, Fhat_woex, Fhat_wex, test]:
        i.index = _set_time_index(end, exg, window, period)
        i.columns = end.columns

    return Fhat_woex_B, Fhat_wex_B, Fhat_woex, Fhat_wex, test, MSE



def _set_time_index(end, exg, window, period):
    """

    :param end: .
    :param exg: .
    :param window:.
    :param period: .
    :param df: dataframe whose index to be changed
    :return:
    """
    var = var_cls.var(
            end,
            exg,
            p = 1
        )

    T_entire = var.end.index

    return   T_entire[window:(window + period)]

def _restore_beta(d_beta_hat_t, beta_t):
    """
    :return: beta_hat_t = beta_t-1 + d_beta_t
    """
    beta_shift = beta_t.shift(1).dropna()

    # get common index
    ind_common = d_beta_hat_t.merge(beta_shift,
                                    right_index=True, left_index=True, how="inner").index

    beta = d_beta_hat_t.loc[ind_common] + beta_shift.loc[ind_common]

    return beta


# def _get_vol_MSE():









