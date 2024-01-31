#####################################
### Produce the data summary table ##
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

options = pd.read_csv("OUTPUT/OP_MOD960104_190626.csv")

# Group indicator
options["exp_days_group"] = options.apply(
    lambda x: "tau <= 30" if (x["days_to_expire"] <= 30) else
    "30 < tau <= 90"  if (30 < x["days_to_expire"]) & (x["days_to_expire"]<= 90) else
    "90 < tau <= 180" if (90 < x["days_to_expire"]) & (x["days_to_expire"]<= 180) else
    "180 < tau <= 365" if (180 < x["days_to_expire"]) & (x["days_to_expire"]<= 365) else
    "365 < tau",
    axis = 1
)

options["log_moneyness_group"] = options.apply(
    lambda x: "M <= -0.2" if (x["log_moneyness"] <= -0.2) else
    "-0.2 < M <= 0"  if (-0.2 < x["log_moneyness"]) & (x["log_moneyness"]<= 0) else
    "0 < M <= 0.2" if (0 < x["log_moneyness"]) & (x["log_moneyness"]<= 0.2) else
    "0.2 < M <= 0.8" if (0.2 < x["log_moneyness"]) & (x["log_moneyness"]<= 0.8) else
    "0.8 < M",
    axis = 1
)
options = options[options["date"] <= "2019-06-26"]

# group by tau
a = options[["impl_volatility", "exp_days_group"]].\
    groupby(["exp_days_group"]).describe(percentiles = [])
a = a.reindex(
    ['tau <= 30', '30 < tau <= 90', '90 < tau <= 180', "180 < tau <= 365", "365 < tau"],
    level = "exp_days_group").T
print(a.to_string())


# group by call/put and log_moneyness
b = options[["impl_volatility", "cp_flag","log_moneyness_group"]].\
    groupby(["cp_flag","log_moneyness_group"]).describe(percentiles = [])
b = b.reindex(
    ['M <= -0.2', '-0.2 < M <= 0', "0 < M <= 0.2", "0.2 < M <= 0.8", "0.8 < M"],
    level = "log_moneyness_group").T
print(b.to_string())
