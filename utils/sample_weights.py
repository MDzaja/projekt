import numpy as np

def get(price_series, trend_labels):
    #create empty numpy arrays for return and duration of each wave
    returnWs = np.array([])
    durationWs = np.array([])

    for i in range(0, len(price_series)-1):
        j = i + 1
        while j < len(price_series)-1 and trend_labels[i] == trend_labels[j]:
            j += 1
        returnWs = np.append(returnWs, abs(price_series[j]/price_series[i]-1))
        durationWs = np.append(durationWs, j-i)
    
    #add mean to both arrays
    returnWs = np.append(returnWs, np.mean(returnWs))
    durationWs = np.append(durationWs, np.mean(durationWs))

    #normalize both arrays to range [0,1]
    returnWs = (returnWs - np.min(returnWs)) / (np.max(returnWs) - np.min(returnWs))
    durationWs = (durationWs - np.min(durationWs)) / (np.max(durationWs) - np.min(durationWs))

    return returnWs

    
