#########################
##### VAR and VAR-X #####
#########################

# TODO: Turn this script into module
# TODO: Hyperparameters: Window size, tickers, lag

# generic packs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time series pack
from statsmodels.tsa.api import VAR

# Read UMC
UMC = pd.read_csv("OUTPUT/Data/UMC.csv", index_col=0)
UMC = UMC.UMC

# 1st diff
Betas = pd.read_csv("OUTPUT/bayes_beta_XOM.csv", index_col=0)

Betas_1 = Betas.diff(1) # 1st order diff
Betas_1 = Betas_1[1:]

# Merge
beta_1_UMC = Betas_1.merge(UMC, how = "inner", left_index = True, right_index = True)
ind = beta_1_UMC.index


# Filtered beta_1 and UMC
Betas_1 = beta_1_UMC.iloc[:,0:5]
UMC = beta_1_UMC.UMC



###### Model 1: VAR(1) ######

# Vault
vault = pd.DataFrame(np.zeros(Betas_1.shape))
vault.index = Betas_1.index
vault.columns = Betas_1.columns

# Rolling forecast
w = 20   # window length
T = len(Betas_1)
for i in range(w, T-1):
    print(i)
    date_p = Betas_1.index[i+1]
    window = Betas_1.iloc[i-w : i]
    model = VAR(window)
    VAR1mode = model.fit(maxlags=1)

    pred = VAR1mode.forecast(window.values[-1:] ,steps=1)
    print(pred[0])
    vault.loc[date_p] = pred[0]

# Add back to original betas
delta_beta_VAR = vault # save

vault = vault.shift(-1)
vault = vault.iloc[w:-1]
Betas = Betas.loc[ind][w:-1]

Betas_hat_VAR = Betas + vault # save



###### Model 2: VAR(1)-X ######

vault = pd.DataFrame(np.zeros(Betas_1.shape))
vault.index = Betas_1.index
vault.columns = Betas_1.columns

def normalize(series):
    mu = series.mean()
    std = series.std()
    return (series - mu) / std

# Rolling forecast
for i in range(w, T-1):
    print(i)
    date_p = Betas_1.index[i+1]
    window = Betas_1.iloc[i-w : i]
    x = UMC.iloc[i-w : i]
    model = VAR(endog = window,
                exog=x)
    VAR1xmode = model.fit(maxlags=1)

    pred = VAR1xmode.forecast(window.values[-1:], exog_future= x.values[-1:], steps=1)
    vault.loc[date_p] = pred[0]

# Add back to beta

# Add back to original betas
delta_beta_VARX = vault # save

vault = vault.shift(-1)
vault = vault.iloc[w:-1]

Betas_hat_VARX = Betas + vault # save



###### Plot: compare VAR and VARX of first difference
for i in vault.columns:
    # plt.margins(x=0)
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)

    # VAR
    ax.plot(delta_beta_VAR[i], alpha = 0.7, label = r"$\Delta$ VAR")

    # VarX
    ax.plot(delta_beta_VARX[i], alpha = 0.7, label = r"$\Delta$ VARX")

    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.show()

###### Plot: compare beta VAR and VARX of first difference
for i in vault.columns:
    # plt.margins(x=0)
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)

    # Raw Beta
    # ax.plot(Betas[i], alpha=0.4, label=r"$\beta$")

    # VAR
    ax.plot(Betas_hat_VAR[i] - Betas[i], alpha=0.7, label=r"$\beta_{VAR}$")

    # VarX
    ax.plot(Betas_hat_VARX[i] - Betas[i], alpha=0.7, label=r"$\beta_{VAR-X}$")

    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.show()


# NSE for VAR and VARX beta from raw beta
rmse_VAR = np.sqrt(np.sum((Betas_hat_VAR - Betas)**2) / 4020)
rmse_VARX = np.sqrt(np.sum((Betas_hat_VARX - Betas)**2) / 4020)
rmse = pd.DataFrame([rmse_VAR, rmse_VARX], index=["VAR", "VARX"])


# Save beta_hat
Betas_hat_VAR.to_csv("OUTPUT/beta_var_XMO.csv")
Betas_hat_VARX.to_csv("OUTPUT/beta_varx_XMO.csv")
