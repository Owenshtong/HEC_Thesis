################################################
###### Vector autoregression time series #######
##### based on  KADIYALA &  KARLSSON (1997) ####
################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.arima.model as ARIMA
from SCRIPT.baysian.baysian_ols.Baysian_OLS import OLS as OLS
from statsmodels.tsa.api import VAR

class var:
    def __init__(self, end, exg, p, pi=None, own_lag_prior_mean = 0):
        """
        Initialize a var object
        :param end: endogeneous variables (y)
        :param exg: exgoeneous variables (x)
        :param p: # of lags in VAR model
        :return: a var object
        """

        if pi is None:
            pi = [0.05, 0.005, 100000]


        self.q = exg.shape[1]  # # exg variables
        self.m = end.shape[1]  # dimension of var

        self.end = pd.DataFrame(end)
        self.end.index = pd.to_datetime(self.end.index)

        self.exg = pd.DataFrame(exg)
        self.exg.index = pd.to_datetime(self.exg.index)


        exg_end = pd.merge(exg, end, right_index=True,left_index=True).dropna()

        self.exg = exg_end.iloc[:,:self.q]
        self.end = exg_end.iloc[:,self.q:]

        # Data config
        self.p = p # #of lags
        self.T = len(exg_end) # time frame length

        # The Minnisota prior
        self.pi1 = pi[0]
        self.pi2 = pi[1]
        self.pi3 = pi[2]

        # 1st lag prior mean of own variable
        self.own_lag_prior_mean = own_lag_prior_mean


    def Z(self, ex = True):
        """
        Build the Z info matrix
        :param ex: True if include exogenous variables
        :return: matrix Z numpy array
        """

        # Without exg
        Z = np.empty([0,self.m * self.p])
        for i in range(self.p, len(self.end)):
            sub_i = np.array(
                self.end.iloc[i-self.p:i,][::-1]
            ).reshape(1, self.p * self.m)
            Z = np.vstack([Z, sub_i])

        # Add exogenous variables
        if ex:
            c = np.array(self.exg.iloc[(self.p - 1):(self.T - 1),])
            Z = np.c_[c, Z]

        # Add unity column for the constant term
        Z = np.c_[np.ones([self.T - self.p, 1]), Z]
        return Z

    def Y(self):
        return np.array(
            self.end.iloc[self.p:,]
        )


    ############# Bayesian ##############
    def Sigma_tilde(self,
                    ex = True):
        """
        :return: Prior var matrix. Diagonal. Follow Litterman (1986) or page 102
        by K & K.
        """

        # The minnisota prior variance
        pi_1 = self.pi1
        pi_2 = self.pi2
        pi_3 = self.pi3

        # Dimension of the factor parameter covariance matrix
        block_dim = self.q + self.m * self.p + 1 if ex else self.m * self.p + 1
        target_dim = self.m * block_dim


        # Step1: AR(p) residual standard error
        def _get_rse(series, p = self.p):
            arp = ARIMA.ARIMA(endog=series,
                              order=(p, 0, 0))
            result = arp.fit()
            return np.sqrt(result.mse)

        sigmas = self.end.apply(_get_rse) # End variables var
        # print(sigmas)

        # Step 2: obtain the diagonal array
        diag = []
        for i in range(self.m): # loop over sigma matrix
            # Initialize
            vault_i = []

            # ith variable var
            sigma_i = sigmas.iloc[i]

            # endo var matrix
            vault_endo = []
            for k in range(self.p): # loop over lags
                vault_k = np.zeros(self.m)
                for j in range(self.m):
                    vault_k[j] = pi_2 * sigma_i**2 / ((k+1)*sigmas[j] **2)
                vault_k[i] = pi_1 / (k+1)
                vault_endo = np.append(vault_endo, vault_k)
            # vault_i = np.append(vault_i, vault_endo)

            # exg var block
            if ex:
                vault_i = np.append(np.repeat(pi_3 * sigma_i**2, self.q + 1),
                                    vault_endo)
            else:
                vault_i = np.append(pi_3 * sigma_i**2,
                                    vault_endo)
            diag = np.append(diag, vault_i)


        Sigma = np.diag(diag)
        print("Shape of prior covariance matrix: " + str(Sigma.shape))
        print("Should be the shape of prior covariance matrix: " + str((target_dim, target_dim)))
        return Sigma


    def gamma_tilde(self, ex = True):
        """
        Prior-mean for gamma. Litterman (1980)
        :return:
        """
        vault = []

        # Prior mean for endo
        for i in range(self.m):
            A1i = np.zeros(self.m)
            A1i[i] = self.own_lag_prior_mean # Note: one of the input when initiating the object
            A1i = np.append(A1i, np.zeros(self.m * (self.p - 1)))
            if ex:
                A1i = np.append(np.zeros(self.q + 1), A1i)
            else:
                A1i = np.append(0, A1i)
            vault = np.append(vault, A1i)

        return vault

    def ols(self, _ex):
        Y_obs = self.Y()
        Z_obs = self.Z(ex = _ex)

        # gamma's parameter OLS
        vault_gamma = []
        # Phi's parameter LS
        vault_Phi = []

        for i in range(self.m):
            y = Y_obs[:,i]
            sig, coef = OLS(y, Z_obs)
            vault_gamma = np.append(vault_gamma, coef)
            vault_Phi = np.append(vault_Phi, sig)


        # The Phi ols estimator serves only as an initial points for MCMC drawing.
        dim = self.q  + self.m * self.p + 1 if _ex else self.m * self.p + 1
        gamma = vault_gamma
        Gamma = vault_gamma.reshape([dim, self.m])
        Phi = np.diag(vault_Phi)

        return gamma, Gamma, Phi


    ############# Forecast ##############
    def _A_D(self, gamma, ex = True):
        """
        Given a vect(Gamma), Build matrix A between (1) and (2). The input gamma should be
        obtained from simulation, or it has to be vect(Gamma)

        :return: matrix big A and D
        """

        # Create a 3d np array vault
        vault_gamma = np.zeros([self.p, self.m, self.m])
        vault_c = np.zeros([self.q + 1, self.m]) if ex else np.zeros([1, self.m])

        # get A1 to Ap, C
        if ex:
            unit_len = self.q + 1 + self.m * self.p
            for i in range(self.m):
                gamma_i = gamma[(i * unit_len): (i+1) * unit_len]
                vault_c[:, i] = gamma_i[:(self.q + 1)]
                vault_gamma[:,:,i] = gamma_i[(self.q + 1):].reshape([self.m, self.p], order = "F").transpose()
        else:
            unit_len = 1 + self.m * self.p
            for i in range(self.m):
                gamma_i = gamma[(i * unit_len): (i+1) * unit_len]
                vault_c[:, i] = gamma_i[:1]
                vault_gamma[:,:,i] = gamma_i[1:].reshape([self.m, self.p], order = "F").transpose()

        # Synthesis big A
        dim_A = [self.m * self.p, self.m * self.p]
        right = np.eye((self.p - 1) * self.m)
        right = np.vstack(
            [right, np.zeros([self.m, (self.p - 1)*self.m])]
        )

        vault_A = vault_gamma[0,:,:]
        for i in range(1,self.p):
            vault_A = np.vstack([vault_A,vault_gamma[i,:,:]])

        Big_A = np.hstack([vault_A, right])


        # Synthesis big D
        if ex:
            Big_D = np.hstack([
                vault_c, np.zeros([self.q + 1, (self.p - 1) * self.m])
            ])
        else:
            Big_D = 0

        # Warning: There might be problem of using 0 * x to get a tiny number.
        return Big_A, Big_D


    def F(self, gamma, h, ex):
        """
        The forecast function (3). We focus on 1-step-ahead
        forecast in our case.
        :return: Big F
        """

        A, D = self._A_D(gamma, ex = ex)

        yt_star = np.array(
            self.end.iloc[-self.p:, :].sort_index(ascending=False)).reshape([1, self.p * self.m])
        if ex:
            xt = np.append(1, np.array(self.exg.iloc[-1:,]))

            sum_DAi = 0
            for i in range(h):
                sum_DAi = sum_DAi + D @ np.linalg.matrix_power(A, i)
            Fh = yt_star @ np.linalg.matrix_power(A, h) + xt @ sum_DAi
        else:
            Fh = yt_star @ np.linalg.matrix_power(A, h)

        return Fh, Fh[:, :self.m].reshape(self.m)

    def var_h_prj(self):
        """
        one step ahead conventional var forecast
        :return:
        """

        # without exgeneous variables
        model = VAR(endog=self.end)
        VAR1xmode = model.fit(maxlags=1)

        pred_woexg = VAR1xmode.forecast(self.end.values[-self.p:], steps=1)

        # with exogenous variables
        model = VAR(endog=self.end, exog=self.exg)
        VAR1xmode = model.fit(maxlags=1)

        pred_wexg = VAR1xmode.forecast(self.end.values[-self.p:],exog_future=self.exg.values[-1:], steps=1)
        return pred_woexg, pred_wexg
