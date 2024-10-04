#############################################################
#################### Auxiliary ##############################
#############################################################

import numpy as np
import pandas as pd

from SCRIPT.Gibbs.gibbs_sampler import gibbs_sampler
from SCRIPT.var import var_cls
import matplotlib.pyplot as plt
from SCRIPT.IV.IV import IV_Remi
import copy
import functools as ft

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



    # Set the correct index
    for i in [Fhat_woex_B, Fhat_wex_B, Fhat_woex, Fhat_wex, test]:
        i.index = _set_time_index(end, exg, window, period)
        i.columns = end.columns

    return Fhat_woex_B, Fhat_wex_B, Fhat_woex, Fhat_wex, test



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

def restore_beta(d_beta_hat_t, beta_t):
    """
    :return: beta_hat_t = beta_t-1 + d_beta_t
    """
    beta_shift = beta_t.shift(1).dropna()

    # get common index
    ind_common = d_beta_hat_t.merge(beta_shift,
                                    right_index=True, left_index=True, how="inner").index

    beta = d_beta_hat_t.loc[ind_common] + beta_shift.loc[ind_common]

    return beta

def merge_and_volhat(option, beta, beta_B, beta_B_UMC, beta_OLS, beta_OLS_UMC):
    """
    1. Merge beta, beta hat (Bayesian), beta hat (bayesian) with UMC.
    2. Add columns of vol hat
    :param option: option data
    :param beta:
    :return:
    """
    option_c = copy.copy(option)
    option_c.date = pd.to_datetime(option_c.date)

    beta_B_c = copy.copy(beta_B)
    beta_B_UMC_c = copy.copy(beta_B_UMC)
    beta_OLS_c = copy.copy(beta_OLS)
    beta_OLS_UMC_c = copy.copy(beta_OLS_UMC)

    # Rename beta_B
    beta_B_c.columns = [i + "_B" for i in beta_B_c.columns]
    beta_B_UMC_c.columns = [i + "_B_UMC" for i in beta_B_UMC_c.columns]
    beta_OLS_c.columns = [i + "_OLS" for i in beta_OLS_c.columns]
    beta_OLS_UMC_c.columns = [i + "_OLS_UMC" for i in beta_OLS_UMC_c.columns]

    # Merge by date
    option_c = ft.reduce(lambda x,y: pd.merge(x, y, right_index=True, left_on='date', how='inner'),
    [option_c, beta, beta_B_c, beta_B_UMC_c, beta_OLS_c, beta_OLS_UMC_c])

    # vol_hat
    IV_vect = np.vectorize(IV_Remi)
    option_c["IV_B"] = IV_vect(
        option_c["log_moneyness"], option_c["tau"],
        option_c["b1_B"], option_c["b2_B"],option_c["b3_B"],option_c["b4_B"],option_c["b5_B"]
    )
    option_c["IV_B_UMC"] = IV_vect(
        option_c["log_moneyness"], option_c["tau"],
        option_c["b1_B_UMC"], option_c["b2_B_UMC"],option_c["b3_B_UMC"],option_c["b4_B_UMC"],option_c["b5_B_UMC"]
    )
    option_c["IV_OLS"] = IV_vect(
        option_c["log_moneyness"], option_c["tau"],
        option_c["b1_OLS"], option_c["b2_OLS"], option_c["b3_OLS"], option_c["b4_OLS"], option_c["b5_OLS"]
    )
    option_c["IV_OLS_UMC"] = IV_vect(
        option_c["log_moneyness"], option_c["tau"],
        option_c["b1_OLS_UMC"], option_c["b2_OLS_UMC"], option_c["b3_OLS_UMC"], option_c["b4_OLS_UMC"], option_c["b5_OLS_UMC"]
    )

    return option_c


def group_indicator(options):
    # Group indicator
    options["exp_days_group"] = options.apply(
        lambda x: "tau <= 30" if (x["days_to_expire"] <= 30) else
        "30 < tau <= 90" if (30 < x["days_to_expire"]) & (x["days_to_expire"] <= 90) else
        "90 < tau <= 180" if (90 < x["days_to_expire"]) & (x["days_to_expire"] <= 180) else
        "180 < tau <= 365" if (180 < x["days_to_expire"]) & (x["days_to_expire"] <= 365) else
        "365 < tau",
        axis=1
    )

    options["log_moneyness_group"] = options.apply(
        lambda x: "M <= -0.2" if (x["log_moneyness"] <= -0.2) else
        "-0.2 < M <= 0" if (-0.2 < x["log_moneyness"]) & (x["log_moneyness"] <= 0) else
        "0 < M <= 0.2" if (0 < x["log_moneyness"]) & (x["log_moneyness"] <= 0.2) else
        "0.2 < M <= 0.8" if (0.2 < x["log_moneyness"]) & (x["log_moneyness"] <= 0.8) else
        "0.8 < M",
        axis=1
    )
    options = options[options["date"] <= "2019-06-26"]

    return options








