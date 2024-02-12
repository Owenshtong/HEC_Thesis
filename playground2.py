import pandas as pd

from SCRIPT.DataPrep import option_cls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BB_XOM = pd.read_csv()

for i in XOM_BB.columns:
    fig, ax = plt.subplots(figsize=(50, 10), dpi = 200)
    ax.plot(XOM_BB[i],label = "CVX")
    ax.plot(XOM.betas[i], label = "XOM")
    plt.margins(x=0)
    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.gcf()
    # plt.savefig("OUTPUT/Plot/baysian_" + i + "_" + ticker + ".png")
    plt.show()

