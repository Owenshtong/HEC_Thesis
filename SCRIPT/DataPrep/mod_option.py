############################################
#######  Build the required dataset ########
############################################

import pandas as pd
import numpy as np

def mod_option(path_fwd,
               path_option,
               ITM: bool = False):
    """
    :param path_fwd: forwrad price data path (input)
    :param path_option: option price data path (input)
    :param ITM: False if exclude ITM options
    :return:
    """
    opt = pd.read_csv(path_option)
    fwd = pd.read_csv(path_fwd)

    # rename to math two dataset
    fwd.rename(columns={"expiration": "exdate"}, inplace=True)
    fwd = fwd.drop(["secid"], axis = 1)
    fwd = fwd.groupby(by = ["date", "exdate"]).mean().reset_index()

    # merge
    opt = opt.merge(fwd, on=['date', 'exdate'], how='left', indicator=True)
    opt = opt[opt['_merge'] == 'both']
    opt = opt.drop(columns='_merge')

    #### Modification ####

    # Strike price: divided by 1000
    opt["strike_price"] = opt["strike_price"] / 1000

    # Annualized date to expire
    opt["days_to_expire"] = opt.apply(lambda x: np.busday_count(x["date"], x["exdate"]), axis=1)
    opt["tau"] = opt["days_to_expire"] / 252

    # Log Moneyness
    opt["log_moneyness"] = (1 / np.sqrt(opt["tau"])) * \
                               np.log(opt["forwardprice"] / opt["strike_price"])

    # Mid-ask and bid (as option price)
    opt["mid_ask_bid"] = (opt["best_offer"] + opt["best_bid"]) / 2


    #### Filtration ####

    # Remove those with less than 6 days to expire
    opt = opt[opt["days_to_expire"] >= 6]

    # Exclude those with price less than 3/8$
    opt = opt[opt["mid_ask_bid"] >= 3 / 8]

    # Exclude 0 bid price
    opt = opt[opt["best_bid"] != 0]

    # Exclude those with a bid-ask spread larger than 175% of the option’s mid-price
    opt = opt[opt["best_offer"] - opt["best_bid"] <= (1.75 * opt["mid_ask_bid"])]

    # Exclude ITM call and put
    if not ITM:
        opt["ITM_indicator"] = opt.apply(lambda x:
                                                 1 if ((x["cp_flag"] == "C") & (x["log_moneyness"] >= 0) |
                                                       (x["cp_flag"] == "P") & (x["log_moneyness"] < 0))
                                                 else 0, axis=1)
        opt = opt[opt["ITM_indicator"] == 0]


    # Exclude those implied volatility are nan
    opt = opt[~np.isnan(opt["impl_volatility"])]


    # Drop unnecessary columns
    opt = opt.drop(["Unnamed: 0_x", "secid"] ,axis = 1)
    return opt
