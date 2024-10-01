import matplotlib.pyplot as plt
from importlib import reload
from SCRIPT.options import option_cls
import numpy as np
import pandas as pd
from SCRIPT.var import var_cls
from SCRIPT.Gibbs.gibbs_sampler import gibbs_sampler as Gibb
from SCRIPT.IV.IV_surface_plt import plt_dyn_surf
import plotly.graph_objects as go




CVX = option_cls.option("CVX", 102968) # Go to WRDS for the id




UMC = pd.read_csv("OUTPUT/UMC.csv", index_col=0)


var = var_cls.var(CVX.betas_d1.iloc[:252], UMC, 3)
a, b = Gibb(var, 2000, ex=False)
a0, b0 = Gibb(var, 2000, ex=True)


vault_noex = []
vault_ex = []
for i in range(2000):
    _, yt1 = var.F(a[i,:], h = 1, ex = False)
    _, yt1_ex = var.F(a0[i,:], h = 1, ex = True)
    vault_noex = np.append(vault_noex, yt1)
    vault_ex = np.append(vault_ex, yt1_ex)

vault_noex = vault_noex.reshape([2000, 5])
vault_ex = vault_ex.reshape([2000, 5])


for i in range(5):
    plt.hist(vault_noex[1000:, i], bins = 50, alpha = 0.5, label= "without UMC")
    plt.hist(vault_ex[1000:, i], bins = 50, alpha = 0.5, label = "UMC")
    # pd.DataFrame(vault_noex).iloc[:, i].plot(kind="density",label= "without UMC")
    # pd.DataFrame(vault_ex).iloc[:, i].plot(kind="density",label= "with UMC")
    plt.legend()
    plt.show()









for i in range(2, 7):
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
    ax.plot(a[200:1000,i])
    ax.plot(a0[200:1000,i+1])
    # ax.plot(a2[200:1000,i])
    plt.margins(x=0)
    plt.show()

for i in range(8, 13):
    plt.hist(a[1000:, i], bins = 60, alpha = 0.5)
    plt.hist(a0[1000:, i+1],bins = 60, alpha = 0.5)
    plt.show()

for i in range(13):
    pd.DataFrame(a).iloc[:, i].plot(kind="density")
    pd.DataFrame(a0).iloc[:, i].plot(kind="density")
    # pd.DataFrame(a2).iloc[2000:, i].plot(kind="density")
    plt.show()

vault1 = []
# vault2 = []
# vault3 = []
for j in range(20):
    for i in range(50, 2000):
        mean1 = np.mean(a[:i, j])
        # mean2 = np.mean(a1[:i, j])
        # mean3 = np.mean(a2[:i, j])
        vault1 = np.append(vault1, mean1)
        # vault2 = np.append(vault2, mean2)
        # vault3 = np.append(vault3, mean3)
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
    ax.plot(vault1)
    # plt.plot(range(10, 10000), vault2)
    # plt.plot(range(10, 10000), vault3)
    plt.show()
    vault1 = []
    # vault2 = []
    # vault3 = []


# F_wo_B, F_w_B, F_wo, F_w
realized = ibm.betas.loc[rb(F_w_B, ibm.betas).index]
beta_hat_B = rb(F_wo_B, ibm.betas)
beta_hat_B_UMC = rb(F_w_B, ibm.betas)



from SCRIPT.IV.IV import IV_Remi
import plotly.io as pio
import plotly.graph_objects as go




t = "2014-04-30"

plt_dyn_surf_smaple(realized.loc[t],
                    cvx.option[cvx.option.date == t ],  t ,
                    Beta_B = beta_hat_B.loc[t ],
                    Beta_B_UMC=beta_hat_B_UMC.loc[t ]
                    )





def _option_beta_merger(option, beta_bay, beta_bay_umc):













