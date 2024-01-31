#########################
##### VAR and VAR-X #####
#########################
import copy

# generic packs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time series pack
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# Bayesian coeff
Betas = pd.read_csv("OUTPUT/bayes_beta_XOM.csv", index_col=0)
model = VAR(Betas)

results = model.fit(maxlags=1)

# Vault
vault = pd.DataFrame(np.zeros(Betas.shape))
vault.index = Betas.index
vault.columns = Betas.columns

# Rolling forecast
T = len(Betas)
for i in range(252, T):
    print(i)
    date_p = Betas.index[i+1]
    window = Betas.iloc[i-100 : i]
    model = VAR(window)
    VAR1mode = model.fit(maxlags=1)

    pred = VAR1mode.forecast(window.values[-1:] ,steps=1)
    print(pred[0])
    vault.loc[date_p] = pred[0]

# Save vault
vault.to_csv("OUTPUT/bayes_beta_XOM_VAR1.csv")

fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
ax.plot(vault["b1"]["2006-09-06":], label = "var")
ax.plot(Betas["b1"]["2006-09-06":], label = "Original")
ax.set_xticks(ax.get_xticks()[::200])
plt.legend()
plt.show()




