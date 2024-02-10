############################################
### Script to build the required dataset ###
############################################

import pandas as pd
import numpy as np



# GLOBAL PARA #
START_DATE = "050901"
END_DATE = "221001"
ticker = "XOM"


# Read in data
options = pd.read_csv(r"INPUT/OptionMetric/Companies/XOM/OP_" + START_DATE + "_" + END_DATE + ".csv")
forward_price = pd.read_csv(r"INPUT/OptionMetric/Companies/XOM/FP_" + START_DATE + "_" + END_DATE + ".csv")
forward_price.rename(columns = {"expiration": "exdate"}, inplace=True)
option_var_to_drop = ["index_flag", "issuer", "exercise_style", "optionid"]  # Options gives na forward prices. Match
                                                                                          # forward price with forward_price by date
forward_price_var_to_drop = ["secid"]


# Drop unnecessary columns
options = options.drop(option_var_to_drop, axis=1)
forward_price = forward_price.drop(forward_price_var_to_drop, axis = 1)
forward_price = forward_price.groupby(by = ["date", "exdate"]).mean().reset_index()


# Get rid of those problematic combinations
options = options.merge(forward_price, on=['date', 'exdate'], how='left', indicator=True)
options = options[options['_merge'] == 'both']
options = options.drop(columns='_merge')


## Add and modification of necessary columns ##

# Strike price: divided by 1000
options["strike_price"] = options["strike_price"]/1000

# Annualized date to expire
options["days_to_expire"] = options.apply(lambda x: np.busday_count(x["date"], x["exdate"]), axis = 1)
options["tau"] = options["days_to_expire"] / 252

# options["days_to_expire"] = pd.to_datetime(options["exdate"]) - pd.to_datetime(options["date"])
# options["days_to_expire"] = options["days_to_expire"].dt.days
# options["tau"] = options["days_to_expire"] / 252

options["days_to_expire"] = options["days_to_expire"] - 1



# Log Moneyness
options["log_moneyness"] = (1 / np.sqrt(options["tau"])) * \
                           np.log(options["ForwardPrice"] / options["strike_price"])

# Mid-ask and bid (as option price)
options["mid_ask_bid"] = (options["best_offer"] + options["best_bid"])/2




## Necessary Filtration: See Page 8 Rémi ##

# Remove those with less than 6 days to expire
options = options[options["days_to_expire"] >= 6]

# Exclude those with price less than 3/8$
options = options[options["mid_ask_bid"] >= 3/8]

# Exclude 0 bid price
options = options[options["best_bid"] != 0]

# Exclude those with a bid-ask spread larger than 175% of the option’s mid-price
options = options[options["best_offer"] - options["best_bid"] <= (1.75 * options["mid_ask_bid"])]

# Exclude ITM call and put
options["ITM_indicator"] = options.apply(lambda x:
                                         1 if ( (x["cp_flag"] == "C") & (x["log_moneyness"] >= 0) |
                                                (x["cp_flag"] == "P") & (x["log_moneyness"] < 0) )
                                         else 0, axis = 1)
options = options[options["ITM_indicator"] == 0]

# Exclude those on day 2006-10-09
options = options[options["date"] != "2006-10-09"]

# Exclude those implied volatility are nan
options = options[~np.isnan(options["impl_volatility"])]

# Try: am_settlement == 1
# options = options[options["am_settlement"] == 1]


# Save the modified option as CSV
options.to_csv("OUTPUT/OP_MOD_" + ticker + "-" + START_DATE + "_" + END_DATE + ".csv", index = False)
