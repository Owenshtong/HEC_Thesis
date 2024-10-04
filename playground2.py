import numpy as np
import pandas as pd
from SCRIPT.options import option_cls
from SCRIPT.var.results_cls import results
import matplotlib.pyplot as plt


# t = "2012-06-22"
#
# plt_dyn_surf(realized.loc[t],
#                     ibm.option[ibm.option.date == t ],  t ,
#                     Beta_B = beta_hat_B.loc[t ],
#                     Beta_B_UMC=beta_hat_B_UMC.loc[t ]
#                     )

ibm = option_cls.option("IBM", 106276)  # Go to WRDS for the id
ibm.betas_d1.index = pd.to_datetime(ibm.betas_d1.index)
ibm.betas.index = pd.to_datetime(ibm.betas.index)


suf = "/Users/tongshihao/Dropbox/Academic/THESIS_HEC/CODE/OUTPUT/Data/OptionMetric/Companies/IBM/D_beta_111/"
F_w, F_w_B, F_wo, F_wo_B = [pd.read_csv(i, index_col=0, parse_dates=True) for i in [suf + j for j in ["F_w.csv","F_w_B.csv", "F_wo.csv", "F_wo_B.csv"]]]
res = results(ibm.option, ibm.betas, F_wo, F_w, F_w_B, F_wo_B)

res.merged.groupby("date").apply(rmse, col_name1 =  "vol_hat_B",  col_name2 =  "impl_volatility")
res.merged.groupby("date").apply(rmse, col_name1 =  "vol_hat_B_UMC",  col_name2 =  "impl_volatility")




fig, ax = plt.subplots(figsize=(60, 15), dpi=200)
ax.plot(P_IV_B, label = "IV_B")
ax.plot(P_IV_B_UMC,  label = "IV_B_UMC")
ax.set_xticks(ax.get_xticks()[::200])
ax.legend()
plt.show()

C_IV_B = IV_B.loc[["C"], :, :].values
P_IV_B = IV_B.loc[["P"], :, :].values
C_IV_B_UMC = IV_B_UMC.loc[:, ["C"], :].values
P_IV_B_UMC = IV_B_UMC.loc[:, ["P"], :].values


fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_B.values, label = "IV_B",alpha = 0.5)
ax.plot(res.IV_rmse_maturity_daily.loc[["tau <= 30"], :, :].IV_B_UMC.values,label = "IV_B_UMC", alpha = 0.5)
plt.legend()
plt.show()