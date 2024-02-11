######## Option class object ########

# 3rd party pack
import pandas as pd
import numpy as np
import os

# my pack
from SCRIPT.DataPrep import wrdsportal, mod_option




class option:
    def __init__(self, ticker: str,
                 secid: int,
                 t0: str = "2003-01-01",
                 tT: str = "2022-08-31",
                 input_path: str = "INPUT/OptionMetric/Companies/",
                 output_path: str = "OUTPUT/OptionMetric/Companies/"):
        self.ticker = ticker
        self.secid = secid
        self.csv_suffix = ticker + "_" + t0[2:4] + t0[5:7] + t0[8:10] + "_" +  tT[2:4] + tT[5:7] + tT[8:10] + ".csv"
        self.intput_path = input_path + ticker
        self.output_path = output_path + ticker
        self.start_date = t0
        self.end_date = tT

        # The merged and filtered final dataset
        self.option = None


    def mod_option(self, itm = False):
        fwd_path = self.intput_path + "/" + "FP_" + self.csv_suffix
        opt_path = self.intput_path + "/" + "OP_" + self.csv_suffix

        # see if the path exists
        if not os.path.exists(self.intput_path):

            # Quote from wrds and make new folder
            wrdsportal.fwd_quote(self.secid, self.start_date, self.end_date, self.ticker)
            wrdsportal.opt_quote(self.secid, self.start_date, self.end_date, self.ticker)

        opt_mod = mod_option.mod_option(
            fwd_path, opt_path, ITM = itm
        )
        self.option = opt_mod
        # TODO: deal with OUTPUT as well. The results so far is good
        # TODO: for some stock the data are too few after filtrating
        return opt_mod




