import pandas as pd

#3-state labeling algorithm
def get_series_labels(series: pd.Series, tau: float=0.05, w: int=11) -> pd.Series:
    first_price = series[0]
    first_trend = None

    for t in range(1, len(series)):
        if series[t] >= first_price + tau*first_price:
            first_trend = 1
            break
        elif series[t] <= first_price - tau*first_price:
            first_trend = -1
            break
        elif t > w:
            first_trend = 0
            break
    
    labels = [first_trend]
    last_upt_p = first_price
    last_upt__t = 0
    trend = first_trend
    for t in range(1, len(series)):
        if trend == 1:
            update = upward_trend(last_upt_p, last_upt__t, series[t], t, tau, w)
        elif trend == 0:
            update = no_action_trend(last_upt_p, last_upt__t, series[t], t, tau, w)
        elif trend == -1:
            update = downward_trend(last_upt_p, last_upt__t, series[t], t, tau, w)

        if update != None:
            trend = update
            last_upt_p = series[t]
            last_upt__t = t
        labels.append(trend)

    labels_series = pd.Series(labels, index=series.index)
    
    return labels_series

def upward_trend(last_upt_price: float, last_upt__time: int, price: float, time: int, tau: float, w: int) -> int:
    if price > last_upt_price:
        return 1
    elif time - last_upt__time > w:
        return 0
    elif price <= last_upt_price - tau*last_upt_price:
        return -1
    return None

def no_action_trend(last_upt_price: float, last_upt__time: int, price: float, time: int, tau: float, w: int) -> int:
    if price >= last_upt_price + tau*last_upt_price:
        return 1
    elif price <= last_upt_price - tau*last_upt_price:
        return -1
    elif time - last_upt__time > w:
        return 0
    return None

def downward_trend(last_upt_price: float, last_upt__time: int, price: float, time: int, tau: float, w: int) -> int:
    if price < last_upt_price:
        return -1
    elif time - last_upt__time > w:
        return 0
    elif price >= last_upt_price + tau*last_upt_price:
        return 1
    return None