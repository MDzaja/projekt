import numpy as np

def get(price_series, trend_labels):
    returnWs = np.array([])
    
    i = 0
    while i < len(price_series)-1:
        j = i + 1
        while j < len(price_series)-1 and trend_labels[i] == trend_labels[j]:
            j += 1
        returnPercentage = abs(price_series[j]/price_series[i]-1)*100
        for _ in range (i, j):
            returnWs = np.append(returnWs, returnPercentage)
        i = j
        
    if trend_labels[-1] == trend_labels[-2]:
        returnWs = np.append(returnWs, returnWs[-2])
    else:
        returnWs = np.append(returnWs, 0)

    return returnWs

    
