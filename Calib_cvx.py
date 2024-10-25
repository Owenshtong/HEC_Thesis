### experiments ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SCRIPT.options import option_cls
from SCRIPT.var.helper import _rmse_b

import plotly.graph_objs as go
import plotly.io as pio
import time


# data
UMC = pd.read_csv("OUTPUT/UMC.csv", index_col=0, parse_dates=True).dropna()


cvx = option_cls.option("CVX", 102968)  # Go to WRDS for the id
cvx.betas_d1.index = pd.to_datetime(cvx.betas_d1.index)
cvx.betas.index = pd.to_datetime(cvx.betas.index)


n = 1000
burnin = 800
window = 100
lag = 3
period = 100
end = cvx.betas_d1
exg = UMC


pi2 = np.linspace(0.0035, 0.00425, 5)
pi1 = np.linspace(1.3, 1.8, 5)
Grid_pi1, Grid_pi2 = np.meshgrid(pi1, pi2)

_rmse_b_vect = np.vectorize(_rmse_b, excluded=["n", "burnin", "window", "lag", "period", "end", "exg", "pi2"])
mse_w, mse_wo, mspe_w, mspe_wo = _rmse_b_vect(Grid_pi1, Grid_pi2, n = n, burnin = burnin, window = window, lag = lag, period = period, end = end, exg = exg)



# Reshape the datafarme
df = pd.DataFrame(np.array([Grid_pi1, Grid_pi2, mse_w, mse_wo, mspe_w, mspe_wo]).reshape(6, -1).T, columns = ["pi1", "pi2", "mse_w", "mse_wo", "mspe_w", "mspe_wo"])
df["Potential_mse"] = df.mse_w < df.mse_wo
df["Potential_mspe"] = df.mspe_w < df.mspe_wo
df["w/wo_mse"] = df.mse_w /df.mse_wo
df["w/wo_mspe"] = df.mspe_w /df.mspe_wo

plt.scatter(df.sort_values(by = "mse_w").iloc[:10,].pi1, df.sort_values(by = "mse_w").iloc[:10,].pi2)
plt.show()

plt.scatter(df[df.Potential_mspe == True].pi1, df[df.Potential_mspe == True].pi2)
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

np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/Grid_pi1", Grid_pi1)
np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/Grid_pi2", Grid_pi2)
np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/mse_w", mse_w)
np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/mse_wo", mse_wo)
np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/mspe_w", mspe_w)
np.save("OUTPUT/hyper_mean_1_pi3_100/cvx/mspe_wo", mspe_wo)
df.to_csv("OUTPUT/hyper_mean_1_pi3_100/cvx/cvx_pi_clib.csv")