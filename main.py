### Main Execution Work Space ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SCRIPT.Interpolator as inp, SCRIPT.Baysian_OLS as bay


# Date range of option data
START_DATE = "050901"
END_DATE = "221001"
ticker = "XOM"


# Read in the modified options data
options = pd.read_csv("OUTPUT/OP_MOD_" + ticker + "_" + START_DATE + "_" + END_DATE + ".csv")
analysis_end = "2022-09-30" # date want to end for analysis

### Part1: Get the daily calibrated coefficients
Days = list(options["date"].unique())
ind_end = Days.index(analysis_end)
b1 = []
b2 = []
b3 = []
b4 = []
b5 = []

# Initialize the beta0 prior
t0 = Days[0]
X, y = bay.regressors(options[options['date'] == t0])
_, beta0  = bay.OLS(y, X)
beta3_t0 = beta0[1]
beta5_t0 = beta0[3]



for t in Days[1:(ind_end + 1)]:

    print(t)

    # Options data at t
    options_t = options[options['date'] == t]

    # Regressor
    X, y = bay.regressors(options_t)

    # OLS sigma estimates
    sigma, _ = bay.OLS(y, X)

    # Build prior
    ATM1month_temp = inp.Interpolate_IV(options_t, log_moneyness=0, maturity=20 / 252)
    if ~np.isnan(ATM1month_temp):
        ATM1month = ATM1month_temp

    ATM1year_temp = inp.Interpolate_IV(options_t)
    if ~np.isnan(ATM1year_temp):
        ATM1year = ATM1year_temp


    beta_prior = bay.beta_prior(ATM1year, ATM1month, beta3_t0, beta5_t0)

    # GLS
    beta_post = bay.Bayesian_GLS_coef(beta_prior, sigma, X, y)

    # Update prior beta3 and 5
    beta3_t0 = beta_post[2,0]
    beta5_t0 = beta_post[4,0]

    # Store the betas
    b1.append(beta_post[0,0])
    b2.append(beta_post[1,0])
    b3.append(beta_post[2,0])
    b4.append(beta_post[3,0])
    b5.append(beta_post[4,0])

# Save the results
bayes_beta = pd.DataFrame([b1,b2,b3,b4,b5]).T
bayes_beta.columns = ["b1", 'b2', "b3", "b4", "b5"]
bayes_beta.index = Days[1:(ind_end + 1)]
# bayes_beta.to_csv("OUTPUT/bayes_beta.csv")
bayes_beta.to_csv("OUTPUT/bayes_beta_" + ticker + ".csv")

for i in bayes_beta.columns:
    fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
    ax.plot(bayes_beta[i],label = i)
    plt.margins(x=0)
    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.gcf()
    plt.savefig("OUTPUT/Plot/baysian_" + i + "_" + ticker + ".png")
    plt.show()

