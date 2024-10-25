### experiments ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SCRIPT.options import option_cls
from SCRIPT.var.helper import _rmse_b
import plotly as py

import plotly.graph_objs as go
import plotly.io as pio



# data
UMC = pd.read_csv("OUTPUT/UMC.csv", index_col=0, parse_dates=True).dropna()


nee = option_cls.option("NEE", 104560)  # Go to WRDS for the id
nee.betas_d1.index = pd.to_datetime(nee.betas_d1.index)
nee.betas.index = pd.to_datetime(nee.betas.index)



n = 1000
burnin = 800
window = 100
lag = 3
period = 100
end = nee.betas_d1
exg = UMC


pi2 = np.linspace(0.003, 0.005, 5)
pi1 =np.linspace(1, 1.4, 5)
Grid_pi1, Grid_pi2 = np.meshgrid(pi1, pi2)

_rmse_b_vect = np.vectorize(_rmse_b, excluded=["n", "burnin", "window", "lag", "period", "end", "exg", "pi2"])
mse_w, mse_wo, mspe_w, mspe_wo = _rmse_b_vect(Grid_pi1, Grid_pi2, n = n, burnin = burnin, window = window, lag = lag, period = period, end = end, exg = exg)



# Reshape the datafrme
df = pd.DataFrame(np.array([Grid_pi1, Grid_pi2, mse_w, mse_wo, mspe_w, mspe_wo]).reshape(6, -1).T, columns = ["pi1", "pi2", "mse_w", "mse_wo", "mspe_w", "mspe_wo"])
df["Potential_mse"] = df.mse_w < df.mse_wo
df["Potential_mspe"] = df.mspe_w < df.mspe_wo
df["w/wo_mse"] = df.mse_w /df.mse_wo
df["w/wo_mspe"] = df.mspe_w /df.mspe_wo

plt.scatter(df.sort_values(by = "mse_w").iloc[:10,].pi1, df.sort_values(by = "mse_w").iloc[:10,].pi2)
plt.show()

plt.scatter(df[df.Potential == True].pi1, df[df.Potential == True].pi2)
plt.show()

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



_srmse_plt_dyn(Grid_pi1, Grid_pi2, mspe_w)


# TODO: add skip paramters into the rolloing_Forecaste helper.
# TODO: make surface plot of rmse for UMC bayesian forecaste (also a curve for rmse p1_only)
# TODO: try to fix pi3 and calibrate for pi2. Think of a way to rationalize this.
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/Grid_pi1", Grid_pi1)
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/Grid_pi2", Grid_pi2)
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/mse_w", mse_w)
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/mse_wo", mse_wo)
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/mspe_w", mspe_w)
np.save("OUTPUT/hyper_mean_1_pi3_100/nee/mspe_wo", mspe_wo)
df.to_csv("OUTPUT/hyper_mean_1_pi3_100/nee/nee_pi_clib.csv")