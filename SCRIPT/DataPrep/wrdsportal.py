################################
### WRDS forward and options ###
################################
import pandas as pd
import os

# Set up connections
import wrds
conn = wrds.Connection()

# Quote data
def sql_string( secid: int, #Look up at [https://wrds-www.wharton.upenn.edu/search/company-search/code-lookup/?product_id=274&attribute_type_map=38|secid,5|ticker,7|cusip
                start_date: str,
                end_date: str,
                wrds_t0: int = 1996,
                wrds_T: int = 2023):
    """
    Return the string to quote forward price and option data from wrds-optionmetrics
    :param secid: ID of stocks. Specifically used in wrds
    :param start_date: period start of interest format "2010-01-31"
    :param end_date: period end of interest
    :param wrds_t0: available period data on wrds
    :param wrds_T: available period data on wrds
    :return:
    """
    # List of price sheet
    frdp_name = []  # Forward
    opprcd_name = []  # Option price
    for t in range(wrds_t0, wrds_T + 1):
        frdp_name.append(
            "fwdprd" + str(t)
        )
        opprcd_name.append(
            "opprcd" + str(t)
        )
    tables = pd.DataFrame([frdp_name, opprcd_name]).T
    tables.columns = ["forward_price", "option_price"]

    # Get the string to quote by sql
    query_fwd = ""
    query_opt = ""

    for i in range(len(tables)):
        name_fwd = tables.forward_price[i]
        name_option = tables.option_price[i]

        # Curret fwd query
        q_fwd_i = "select secid, date as date, expiration, forwardprice" + " " \
                                                                           " from optionm." + name_fwd + " " + \
                  " where secid = " + str(secid) + \
                  " and date >= " + "'" + start_date + "'" + \
                  " and date <= " + "'" + end_date + "'"
        if i != len(tables) - 1:
            q_fwd_i = q_fwd_i + " union"
        query_fwd = query_fwd + " " + q_fwd_i

        # Current option query
        q_opt_i = "select secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, impl_volatility" + " " \
                                                                                                               " from optionm." + name_option + " " + \
                  " where secid = " + str(secid) + \
                  " and date >= " + "'" + start_date + "'" + \
                  " and date <= " + "'" + end_date + "'"
        if i != len(tables) - 1:
            q_opt_i = q_opt_i + " union"
        query_opt = query_opt + " " + q_opt_i

    return query_fwd, query_opt


def fwd_quote(secid: int, #Look up at [https://wrds-www.wharton.upenn.edu/search/company-search/code-lookup/?product_id=274&attribute_type_map=38|secid,5|ticker,7|cusip
              start_date: str,
              end_date: str,
              ticker: str):
    sql, _ = sql_string(secid, start_date, end_date)
    fwd = conn.raw_sql(sql)

    # Write to path
    newpath = "INPUT/OptionMetric/Companies/" + ticker
    yyyy0 = start_date[2:4]
    month0 = start_date[5:7]
    day0 = start_date[8:10]
    yyyyT = end_date[2:4]
    monthT = end_date[5:7]
    dayT = end_date[8:10]

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    fwd.to_csv(newpath + "/FP_" + ticker + "_" + yyyy0 + month0 + day0 + "_" + yyyyT + monthT + dayT + ".csv")

    return conn.raw_sql(sql)

def opt_quote(secid: int, #Look up at [https://wrds-www.wharton.upenn.edu/search/company-search/code-lookup/?product_id=274&attribute_type_map=38|secid,5|ticker,7|cusip
              start_date: str,
              end_date: str,
              ticker: str):
    _, sql = sql_string(secid, start_date, end_date)
    opt = conn.raw_sql(sql)

    # Write to path
    newpath = "INPUT/OptionMetric/Companies/" +  ticker
    yyyy0 = start_date[2:4]
    month0 = start_date[5:7]
    day0 = start_date[8:10]
    yyyyT = end_date[2:4]
    monthT = end_date[5:7]
    dayT = end_date[8:10]

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    opt.to_csv(newpath + "/OP_" + ticker + "_" + yyyy0 + month0 + day0 + "_" + yyyyT + monthT + dayT + ".csv")

    return opt

