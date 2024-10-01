######## Option class object ########
import copy

# 3rd party pack
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# my pack
from SCRIPT.wrds import wrdsportal
from SCRIPT.options import mod_option
import SCRIPT.baysian.baysian_ols.Interpolator as inp, SCRIPT.baysian.baysian_ols.Baysian_OLS as bay

#############################################################
#################### Option object class ####################
#############################################################

class option:
    def __init__(self, ticker: str,
                 secid: int,
                 t0: str = "2012-01-01",
                 tT: str = "2018-12-30",
                 input_path: str = "INPUT/OptionMetric/Companies/",
                 output_path: str = "OUTPUT/Data/OptionMetric/Companies/"):
        self.ticker = ticker
        self.secid = secid
        self.csv_suffix = ticker + "_" + t0[2:4] + t0[5:7] + t0[8:10] + "_" +  tT[2:4] + tT[5:7] + tT[8:10] + ".csv"
        self.intput_path = input_path + ticker # folder path
        self.output_path = output_path + ticker  # folder path
        self.start_date = t0
        self.end_date = tT

        # The merged and filtered final dataset
        self.option = self.mod_option()

        # Bayes OLS betas
        self.betas = self.BGLS_betas()
        self.BGLS_Prior_var = None

        # Delta beta
        self.betas_d1 = self.betas.diff(1)[1:]


    def mod_option(self, itm = False):
        # Input csv
        fwd_path = self.intput_path + "/" + "FP_" + self.csv_suffix
        opt_path = self.intput_path + "/" + "OP_" + self.csv_suffix

        # Output csv
        mod_opt_path = self.output_path + "/" + "OP_mod_" + self.csv_suffix


        # Check if output already has the filtered dataset
        if os.path.exists(self.output_path):
            opt_mod = pd.read_csv(mod_opt_path, index_col=0)
            self.option = opt_mod


        # see if the path exists
        else:
            # Quote from wrds and make new folder
            wrdsportal.fwd_quote(self.secid, self.start_date, self.end_date, self.ticker)
            wrdsportal.opt_quote(self.secid, self.start_date, self.end_date, self.ticker)

            opt_mod = mod_option.mod_option(
                fwd_path, opt_path, ITM = itm
            )
            self.option = opt_mod

            # Create repo
            os.makedirs(self.output_path)
            opt_mod.to_csv(mod_opt_path)

        # NOTE: for some stock the data are too few after filtrating
        return opt_mod

    def BGLS_betas(self):
        """
        Run Bayesian OLS on each single betas.
        :return: dataframe of raw betas
        """
        # Check existence
        beta_csv_path = self.output_path + "/BB_" + self.csv_suffix

        if os.path.exists(beta_csv_path):
            bayes_beta = pd.read_csv(beta_csv_path, index_col=0)
            self.betas = bayes_beta

            # Delta beta
            d1 = bayes_beta.diff(1)
            self.betas_d1 = d1[1:]
        else:
            ### Part: get the hyper-paramter (Variance of prior)
            prior_var = bay.__priorVar__(self.option)
            self.BGLS_Prior_var = prior_var

            ### Part1: Get the daily calibrated coefficients
            Days = sorted(self.option["date"].unique())
            b1 = []
            b2 = []
            b3 = []
            b4 = []
            b5 = []

            # Initialize the beta0 prior
            t0 = Days[0]
            X, y = bay.regressors(self.option[self.option['date'] == t0])
            _, beta0 = bay.OLS(y, X)
            beta3_t0 = beta0[1]
            beta5_t0 = beta0[3]

            # Daily calibration
            ATM1month = 0

            for t in Days[1:(len(Days) + 1)]:

                print(t)

                # Options data at t
                options_t = self.option[self.option['date'] == t]

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
                beta_post = bay.Bayesian_GLS_coef(beta_prior, sigma, X, y, prior_var=prior_var)

                # Update prior beta3 and 5
                beta3_t0 = beta_post[2, 0]
                beta5_t0 = beta_post[4, 0]

                # Store the betas
                b1.append(beta_post[0, 0])
                b2.append(beta_post[1, 0])
                b3.append(beta_post[2, 0])
                b4.append(beta_post[3, 0])
                b5.append(beta_post[4, 0])

            # Save the results
            bayes_beta = pd.DataFrame([b1, b2, b3, b4, b5]).T
            bayes_beta.columns = ["b1", 'b2', "b3", "b4", "b5"]
            bayes_beta.index = Days[1:(len(Days) + 1)]

            bayes_beta.to_csv(beta_csv_path)

            self.betas = bayes_beta

        return bayes_beta


    def plot_betas(self, save_fig = False):
        # TODO: move the plot function from main.py to here.
        fig_path = "OUTPUT/Plot/" + self.ticker
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        for i in self.betas.columns:
            fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
            ax.plot(self.betas[i], label=i)
            plt.margins(x=0)
            plt.legend()
            ax.set_xticks(ax.get_xticks()[::200])
            plt.gcf()
            if save_fig:
                plt.savefig(fig_path + "/beta_" + self.ticker)
            plt.show()

    def plot_betas_d1(self, save_fig = False):
        # TODO: move the plot function from main.py to here.
        fig_path = "OUTPUT/Plot/" + self.ticker
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        for i in self.betas.columns:
            fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
            ax.plot(self.betas_d1[i], label=i)
            plt.margins(x=0)
            plt.legend("$\pi_1 = $")
            ax.set_xticks(ax.get_xticks()[::200])
            plt.gcf()
            if save_fig:
                plt.savefig(fig_path + "/beta_" + self.ticker + "d1")
            plt.show()

















