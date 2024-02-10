#######################################
####### UMC series construction #######
#######    Ardia et al. 2022    #######
#######################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as ARIMA

# MCCC index
MCCC_daily = pd.read_excel("INPUT/MCCC.xlsx", sheet_name='2023 update daily', skiprows = 6, index_col=0)
MCCC_agg_daily = MCCC_daily["Aggregate"]
MCCC_agg_daily = MCCC_agg_daily.asfreq('D')


###### Rolling forecast
#   - 1 day ahead
#   - 1000 windows
T = len(MCCC_agg_daily)

# The vault
vault = pd.DataFrame([MCCC_agg_daily, np.repeat(np.nan, T),np.repeat(np.nan, T)]).T
vault.columns = ["MCCC", "MCCC_p","UMC"]

# Rolling
for i in range(1000, T):
    print(i)
    window = MCCC_agg_daily[i-1000 : i]
    ar1mod = ARIMA.ARIMA(endog = window,
               order = (1,0,0))
    mod = ar1mod.fit(method = "hannan_rissanen")
    pred = mod.forecast(steps=1) # predicted
    vault["MCCC_p"].loc[pred.index] = float(pred.iloc[0])

# UMC
vault.UMC = vault.MCCC - vault.MCCC_p

# Plot
fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
ax.plot(vault.MCCC, label = "MCCC")
ax.plot(vault.MCCC_p, label = "Predicted")
plt.legend()
plt.gcf()
plt.savefig("OUTPUT/Plot/MCCC_MCCCp" + ".png")
plt.show()

fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
ax.plot(vault.UMC, label = "UMC")
plt.legend()
plt.gcf()
plt.savefig("OUTPUT/Plot/UMC" + ".png")
plt.show()

# Save UMC index
vault.UMC.to_csv("OUTPUT/UMC.csv")

