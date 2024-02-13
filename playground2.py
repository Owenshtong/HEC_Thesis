import pandas as pd

from SCRIPT.DataPrep import option_cls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SCRIPT.DataPrep import  option_cls

CVX = option_cls.option("CVX", 1000)
CVX.mod_option()
CVX.BGLS_betas()
CVX.plot_betas()

XOM = option_cls.option("XOM", 1000)
XOM.mod_option()
XOM.BGLS_betas()
XOM.plot_betas()

CVX.BGLS_Prior_var
XOM.BGLS_Prior_var