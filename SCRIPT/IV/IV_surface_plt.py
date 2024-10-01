#######################################
### Producing necessary option plot ###
#######################################

# Numerical requirement
import numpy as np
from SCRIPT.IV.IV import IV_Remi

# 3d Plot
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go



def plt_static_surf(betas, option_data, M_u = 2, M_l = -1, tu_u = 0, tu_l = 3, N = 50 ,with_scatter = True, resolution = 500):
    '''
    Returns the static 3d plot together with model IV surface and observed IV. Interpolation is built inside
    '''

    # Surface construction
    M_range = np.linspace(M_l, M_u, N)
    tau_range = np.linspace(tu_l, tu_u, N)
    M, TAU = np.meshgrid(M_range, tau_range)
    IV_vect = np.vectorize(IV_Remi, excluded=["beta"]) # vectorize the IV function
    iv_fit = IV_vect(M, TAU, beta = betas)

    # Plt
    fig = plt.figure(dpi = resolution)
    ax = plt.axes(projection='3d')
    ax.plot_surface(M,TAU, iv_fit, rstride=1, cstride=1)

    if with_scatter:
        ax.scatter3D(option_data["log_moneyness"], option_data["tau"], option_data["impl_volatility"], s = 2, edgecolor= "orange")
    ax.view_init(20, -170)

    plt.show()

    return fig


def plt_dyn_surf(betas, option_data, t, Beta_B=None, Beta_B_UMC=None, M_u=2, M_l=-1, tu_u=0.01, tu_l=3, N=50):
    '''
       Returns the interactive 3d plot together with model IV surface and observed IV. Interpolation is built inside
       t: date. String. For plot title solely.
    '''

    # Settings to show plot in browser
    pio.renderers.default = "browser"

    # Surface construction
    M_range = np.linspace(M_l, M_u, N)
    tau_range = np.linspace(tu_l, tu_u, N)

    # Grid
    M, TAU = np.meshgrid(M_range, tau_range)
    IV_vect = np.vectorize(IV_Remi, excluded=["beta"])  # vectorize the IV function
    iv_fit = IV_vect(M, TAU, beta=betas)


    # Plot

    layout = go.Layout(scene=dict(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.7)
    ))
    fig = go.Figure(data=[go.Surface(x=M, y=TAU, z=iv_fit, colorscale='Blues')], layout=layout)

    if Beta_B is not None:
        iv_fit_beta_B = IV_vect(M, TAU, beta=Beta_B)
        fig = go.Figure(data=[
            go.Surface(x=M, y=TAU,z=iv_fit, colorscale='Blues'),
            go.Surface(x=M, y=TAU,z=iv_fit_beta_B, colorscale='Oranges', showscale=False)
        ],layout = layout)

    if Beta_B_UMC is not None:
        iv_fit_beta_B = IV_vect(M, TAU, beta=Beta_B)
        iv_fit_beta_B_UMC = IV_vect(M, TAU, beta=Beta_B_UMC)
        fig = go.Figure(data=[
            go.Surface(x=M, y=TAU,z=iv_fit, colorscale='Blues'),
            go.Surface(x=M, y=TAU,z=iv_fit_beta_B, colorscale='Oranges',showscale=False),
            go.Surface(x=M, y=TAU,z=iv_fit_beta_B_UMC, colorscale='Greens',showscale=False)
        ], layout = layout)

    if (Beta_B_UMC is not None) & (Beta_B is not None):
        iv_fit_beta_B = IV_vect(M, TAU, beta=Beta_B)
        iv_fit_beta_B_UMC = IV_vect(M, TAU, beta=Beta_B_UMC)
        fig = go.Figure(data=[
            go.Surface(x=M, y=TAU, z=iv_fit_beta_B, colorscale='Oranges', showscale=False),
            go.Surface(x=M, y=TAU, z=iv_fit_beta_B_UMC, colorscale='Greens', showscale=False)
        ], layout=layout)

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=False))

    fig.add_scatter3d(x=option_data.log_moneyness, y=option_data.tau, z=option_data.impl_volatility, mode='markers',
                      marker=dict(size=2,
                                  color='blue'))
    fig.update_scenes(yaxis_autorange="reversed")

    fig.update_layout(
        title='Fitted IV surface on' + t,
        scene=dict(
            xaxis_title='Log Moneyness (M)',
            yaxis_title=r'Maturity $\tau$',
            zaxis_title='Implied Volatility (IV)',
        ))

    return fig


def _3d_scatter(option_data,ang_vert = 20,ang_hor = -170 ,resolution = 500, size = 1):
    fig = plt.figure(dpi=resolution)
    ax = plt.axes(projection='3d')
    ax.scatter3D(option_data["tau"], option_data["log_moneyness"], option_data["impl_volatility"],
                 s=size, facecolor="none", edgecolor="b")
    ax.view_init(ang_vert, ang_hor)
    plt.show()
    return fig

def _tau_vs_moneyness(option_data, date):
    '''
    :param option_data: pd.df, entire option data frame across time
    :param date: string of form "2020-01-12"
    '''
    plt.scatter(option_data[option_data["date"] == date].log_moneyness,
                option_data[option_data["date"] == date].tau)
    plt.title(date)
    plt.xlabel("Log-moneyness")
    plt.ylabel("Maturity")
    plt.show()







