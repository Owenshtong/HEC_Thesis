import numpy as np
import pandas as pd

from SCRIPT.IV.IV_surface_plt import plt_dyn_surf
from SCRIPT.options import option_cls
from SCRIPT.var.results_cls import results
from SCRIPT.var.var_cls import var
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
import time
from SCRIPT.Gibbs.gibbs_sampler import gibbs_sampler
import importlib
import sys, os


ibm = option_cls.option("IBM", 106276)  # Go to WRDS for the id
ibm.betas_d1.index = pd.to_datetime(ibm.betas_d1.index)
ibm.betas.index = pd.to_datetime(ibm.betas.index)


suf = "/Users/tongshihao/Dropbox/Academic/THESIS_HEC/CODE/OUTPUT/Data/OptionMetric/Companies/NEE/D_beta_111/"
F_w, F_w_B, F_wo, F_wo_B = [pd.read_csv(i, index_col=0, parse_dates=True) for i in [suf + j for j in ["F_w.csv","F_w_B.csv", "F_wo.csv", "F_wo_B.csv"]]]
r = pd.read_csv("INPUT/r.csv", index_col=0, parse_dates=True)


res = results(ibm.option, ibm.betas, F_wo, F_w, F_w_B, F_wo_B)





fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_OLS.values, label = "IV_B",alpha = 0.8, linewidth = 0.7)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_OLS_UMC.values,label = "IV_B_UMC", alpha = 0.8, linewidth = 0.7)

ax.plot(
    (res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_OLS.values - res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_OLS_UMC.values) / (res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_OLS.values * 100)
, linewidth = 0.7)
plt.legend()
plt.show()

t = "2015-01-20"


realized = res.raw_beta
beta_hat_B = res.beta_B
beta_hat_B_UMC = res.beta_B_UMC
plt_dyn_surf(realized.loc[t],
                    ibm.option[(ibm.option.date == t) & (ibm.option.cp_flag == "P")],  t ,
                    Beta_B = beta_hat_B.loc[t ],
                    Beta_B_UMC=beta_hat_B_UMC.loc[t ]
                    )


#     os.makedirs(fig_path)

fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
ax.plot(res.O_arpe_daily.O_B, alpha = 0.7, label="Bayesian")
ax.plot(res.O_arpe_daily.O_B_UMC, alpha = 0.7, label="Bayesian UMC")
plt.margins(x=0)
plt.legend()
ax.set_xticks(ax.get_xticks()[::200])
plt.gcf()
# if save_fig:
#     plt.savefig(fig_path + "/beta_" + self.ticker)
plt.show()


def _accum_mean(gamma_df):
    return gamma_df.expanding().mean()



red = pd.read_csv("OUTPUT/hyper_mean_0_pi3_0.8/cvx/cvx_pi_clib.csv")
gridpi1 = np.load("OUTPUT/hyper_mean_0_pi3_0.8/cvx/Grid_pi1.npy")
gridpi2 = np.load("OUTPUT/hyper_mean_0_pi3_0.8/cvx/Grid_pi2.npy")
res_w = np.load("OUTPUT/hyper_mean_0_pi3_0.8/cvx/res_w.npy")

def _srmse_plt_dyn(X_grid, Y_grid, smse_with_umc):
    pio.renderers.default = "browser"
    lines = []
    line_marker = dict(color='#000000', width=4)
    for i, j, k in zip(X_grid, Y_grid, smse_with_umc):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
    for i, j, k in zip(X_grid.T, Y_grid.T, smse_with_umc.T):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
    fig = go.Figure(lines)
    fig.update_scenes(xaxis_autorange="reversed", yaxis_autorange = "reversed")

    return fig

import plotly.graph_objs as go
import plotly.io as pio
import time

_srmse_plt_dyn(gridpi1, gridpi2, res_w)
