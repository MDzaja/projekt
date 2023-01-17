import pandas as pd

def get(price_series: pd.Series, trend_labels: pd.Series) -> pd.Series:
    weight_list = []
    
    i = 0
    while i < len(price_series)-1:
        j = i + 1
        while j < len(price_series)-1 and trend_labels[i] == trend_labels[j]:
            j += 1
        for k in range (i, j):
            returnPercentage = abs(price_series[j]/price_series[k]-1)*100
            weight_list.append(returnPercentage)
        i = j
        
    if trend_labels[-1] == trend_labels[-2]:
        weight_list.append(weight_list[-2])
    else:
        weight_list.append(0)

    return pd.Series(weight_list, index=price_series.index)

    
