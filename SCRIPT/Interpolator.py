### Interpolator for ATM 1-year to maturity option Volatility ###

import scipy as spy
import numpy as np

def Interpolate_IV(options_t, log_moneyness = 0, maturity = 1):
    # options: options data at day t
    # return: the IV interpolated

    # options_t = options_t[
    #     (options_t["tau"] <= 2) &
    #     (options_t["tau"] >= 0.6) &
    #     (options_t["log_moneyness"] <= 0.5) &
    #     (options_t["log_moneyness"] >= -0.5)
    #     ]

    points_ob = options_t[["log_moneyness", "tau"]]
    iv_ob = options_t["impl_volatility"]

    grid_M, grid_tau = np.array([log_moneyness, maturity]).reshape([2, 1, 1])

    grid_z0 = spy.interpolate.griddata(
        points=points_ob,
        values=iv_ob,
        xi=(grid_M, grid_tau),
        method="linear"
    )
    return grid_z0[0,0]




