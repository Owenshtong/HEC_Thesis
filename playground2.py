import numpy as np
import pandas as pd

from SCRIPT.IV.IV_surface_plt import plt_dyn_surf
from SCRIPT.options import option_cls
from SCRIPT.var.results_cls import results
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm


ibm = option_cls.option("IBM", 106276)  # Go to WRDS for the id
ibm.betas_d1.index = pd.to_datetime(ibm.betas_d1.index)
ibm.betas.index = pd.to_datetime(ibm.betas.index)


suf = "/Users/tongshihao/Dropbox/Academic/THESIS_HEC/CODE/OUTPUT/Data/OptionMetric/Companies/IBM/D_beta_111_1/"
F_w, F_w_B, F_wo, F_wo_B = [pd.read_csv(i, index_col=0, parse_dates=True) for i in [suf + j for j in ["F_w.csv","F_w_B.csv", "F_wo.csv", "F_wo_B.csv"]]]
r = pd.read_csv("INPUT/r.csv", index_col=0, parse_dates=True)
res = results(ibm.option, ibm.betas, F_wo, F_w, F_w_B, F_wo_B, r)



fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_B.values, label = "IV_B",alpha = 0.8)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_B_UMC.values,label = "IV_B_UMC", alpha = 0.8)
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


