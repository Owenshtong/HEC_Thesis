################################################
########## Equation 16(a) and 16(b) ############
###########   KADIYALA &  KARLSSON (1997) ######
################################################

import numpy as np
import pandas as pd
from scipy.stats import invwishart as iwish, multivariate_normal as mvnorm
from numpy.linalg import inv
from numpy import kron
from SCRIPT.var import var_cls
import matplotlib.pyplot as plt

def gibbs_sampler(var,n, ex = True):
    """
    Draw samples from 16(a) and 16(b)
    :param ex: including exogenous variables?
    :param var: a var class object
    :param n: number of trails of the chain
    ;:param var: a var class
    :return: dataframe of the beta chain
    """
    # Dimension of the chain dataframe
    dim_beta = [n, (var.q + var.m * var.p + 1) * var.m] if ex else [n, (var.m * var.p + 1) * var.m]
    dim_phi = [n, var.m**2]

    # helper quantity
    def gamma_bar(Z, Phi, SigmaTilde, gammaHat, gammaTilde):
        rep = kron(inv(Phi), Z.T @ Z)
        return inv(
            inv(SigmaTilde) + rep
        ) @ (
            inv(SigmaTilde) @ gammaTilde + rep @ gammaHat
        )

    # Info data
    Z = var.Z(ex = ex)
    Y = var.Y()
    Sigma_tilde = var.Sigma_tilde(ex = ex)
    gamma_hat, Gamma_hat, Phi = var.ols(_ex = ex) # Phi initial
    gamma_tilde = var.gamma_tilde(ex = ex)


    # Save samples to list
    vault_gamma = []
    vault_phi = []
    for i in range(n):

        # repetitive quantity
        rep = kron(inv(Phi), Z.T @ Z)

        # draw from 16(a)
        gammaBar = gamma_bar(
            Z, Phi, Sigma_tilde, gamma_hat, gamma_tilde
        )
        mnormVar = inv(inv(Sigma_tilde) + rep)

        gamma_sample = mvnorm.rvs(gammaBar, mnormVar)
        Gamma_sample = np.reshape(gamma_sample, [var.q + var.m * var.p + 1, var.m]) if ex \
            else np.reshape(gamma_sample, [var.m * var.p + 1, var.m])
        vault_gamma = np.append(vault_gamma, gamma_sample)

        # draw from 16(b)
        rep = Y - Z @ Gamma_hat
        rep_2 = Z @ (Gamma_sample - Gamma_hat)
        scale_mat = inv(
            rep.T @ rep  + rep_2.T @ rep_2
        )

        iwish_sample = iwish.rvs(
            var.T - var.p,
            scale_mat)
        wish_sample = inv(iwish_sample)
        Phi = wish_sample
        vault_phi = np.append(vault_phi,
            np.reshape(wish_sample, [1, var.m**2]) # by column or row are the same since symmetric
        )

    # return the 2 sets of sample
    gamma_samples = vault_gamma.reshape(dim_beta)
    Phi_samples = vault_phi.reshape(dim_phi)

    return gamma_samples, Phi_samples


def _accum_mean(gamma_df):
    return gamma_df.expanding().mean()










