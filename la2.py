#2-state labeling algorithm
def get_series_labels(series, tau=0.15):
    first_price = series[0]
    high_peak_p = series[0]
    high_peak_t = 0
    low_peak_p = series[0]
    low_peak_t = 0
    trend = 0
    first_peak_t = 0

    for i in range(1, len(series)):
        if series[i] > first_price + first_price*tau:
            high_peak_p = series[i]
            high_peak_t = i
            first_peak_t = i
            trend = 1
            break
        if series[i] > first_price - first_price*tau:
            low_peak_p = series[i]
            low_peak_t = i
            first_peak_t = i
            trend = -1
            break
    
    labels = [0]*len(series)
    for i in range(first_peak_t+1, len(series)):
        if trend > 0:
            if series[i] > high_peak_p:
                high_peak_p = series[i]
                high_peak_t = i
            if series[i] < high_peak_p-high_peak_p*tau and low_peak_t <= high_peak_t:
                for j in range(0, len(series)):
                    if j > low_peak_t and j <= high_peak_t:
                        labels[j] = 1
                low_peak_p = series[i]
                low_peak_t = i
                trend = -1
        if trend < 0:
            if series[i] < low_peak_p:
                low_peak_p = series[i]
                low_peak_t = i
            if series[i] > low_peak_p+low_peak_p*tau and high_peak_t <= low_peak_t:
                for j in range(0, len(series)):
                    if j > high_peak_t and j <= low_peak_t:
                        labels[j] = -1
                high_peak_p = series[i]
                high_peak_t = i
                trend = 1

    last_peak = high_peak_t
    if trend > 0:
        last_peak = low_peak_t
    for i in range(last_peak+1, len(series)):
        labels[i] = trend
        
    return labels
