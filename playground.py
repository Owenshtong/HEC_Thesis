### experiments ###
import SCRIPT.Plot_functions as plt_f
import pandas as pd
import plotly
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt
import statsmodels.api as sm
pio.renderers.default = "browser"


options = pd.read_csv("OUTPUT/OP_MOD960104_191231.csv")
bayes_beta = pd.read_csv("OUTPUT/bayes_beta.csv", index_col=0)


# t = "2006-05-08"
# options_t = options[options["date"] == t]
# beta_t = bayes_beta.loc[t]
# a = plt_f.plt_dyn_surf(beta_t,options_t, t)
# plotly.offline.plot(a, filename = t + '.html', auto_open=False)
#
# dif_1 = pd.DataFrame(np.diff(bayes_beta, axis=0))
# dif_1.columns = bayes_beta.columns
# for i in dif_1.columns:
#     fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
#     ax.plot(dif_1[i],label = i)
#     plt.margins(x=0)
#     plt.legend()
#     ax.set_xticks(ax.get_xticks()[::200])
#     plt.gcf()
#     plt.savefig("OUTPUT/Plot/baysian_" + i + "diff1.png")
#     plt.show()
#
# sm.graphics.tsa.plot_acf(dif_1["b5"])
# plt.show()

days = bayes_beta.index
days_sub = days[3200:3300]
for t in days_sub:
    print(t)
    beta_t = bayes_beta.loc[t]
    options_t = options[options["date"] == t]
    a = plt_f.plt_dyn_surf(beta_t, options_t, t, M_u = 3, M_l = -2.1)
    plotly.offline.plot(a, filename = t + '.html', auto_open=False)
