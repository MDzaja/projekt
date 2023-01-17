#create a list of indices for cross validation, each element has 8 train folds and 2 test folds
#after each test fold there is a skip period where no data is used
def custom_ten_fold_cv_selection(data_size: int, window_size: int) -> list:
    unsortedCVindices = []
    sortedByTestCVindices = []
    uniqueCVindices = []

    fold_length = int(data_size/10)
    half_observation_period = int(window_size/2)
    for i in range(0, 10):
        for j in range(i+1, 10):
            trainIndices = []
            testIndices = []

            end_index = (j+1)*fold_length
            if j == 9:
                end_index = data_size

            if j-i == 1:
                for k in range(0, i*fold_length):
                    trainIndices.append(k)
                for k in range(i*fold_length, end_index):
                    testIndices.append(k)
                unsortedCVindices.append((trainIndices, testIndices))
            else:
                for k in range(0, i*fold_length):
                    trainIndices.append(k)
                for k in range(i*fold_length, (i+1)*fold_length-half_observation_period):
                    testIndices.append(k)
                unsortedCVindices.append((trainIndices, testIndices))

                trainIndices = []
                testIndices = []
                for k in range((i+1)*fold_length+window_size-half_observation_period, j*fold_length):
                    trainIndices.append(k)
                for k in range(j*fold_length, end_index):
                    testIndices.append(k)
                unsortedCVindices.append((trainIndices, testIndices))

    for fold in range(1, 10):
        startIndex = fold*fold_length
        for indices in unsortedCVindices:
            if indices[1][0] == startIndex:
                sortedByTestCVindices.append(indices)
    
    for i in range(0, len(sortedByTestCVindices)):
        counter = 0
        for j in range(0, len(sortedByTestCVindices)):
            indicesList = sortedByTestCVindices
            if indicesList[i][0][0] == indicesList[j][0][0] and indicesList[i][0][-1] == indicesList[j][0][-1] \
            and indicesList[i][1][0] == indicesList[j][1][0] and indicesList[i][1][-1] == indicesList[j][1][-1]:
                counter += 1
                if counter > 1:
                    break
        if counter == 1:
            uniqueCVindices.append(sortedByTestCVindices[i])
            
    return uniqueCVindices

#TODO - potentially modify for class number grater than 2
def remove_monoton_instances(cv_indices: list, labels: list) -> list:
    finalCVindices = []
    for indices in cv_indices:
        counter = 0
        for i in [0, 1]:
            for j in range(1, len(indices[i])):
                if labels[indices[i][j]] != labels[indices[i][j-1]]:
                    counter += 1
                    break
        if counter == 2:
            finalCVindices.append(indices)

    return finalCVindices

#create a list of indices for cross validation, each element has 8 train folds and 2 test folds
#after each test fold there is a skip period where no data is used
def simple_ten_fold_cv_selection(data_size: int, window_size: int) -> list:
    customCVindices = []
    fold_length = int(data_size/10)
    half_observation_period = int(window_size/2)
    for i in range(0, 10):
        for j in range(i+1, 10):
            trainIndices = []
            testIndices = []

            end_index = (j+1)*fold_length
            if j == 9:
                end_index = data_size
            for k in range(0, data_size):
                if j-i != 1 and k >= (i+1)*fold_length-half_observation_period and k < (i+1)*fold_length+window_size-half_observation_period:
                    continue
                if j != 9 and k >= (j+1)*fold_length-half_observation_period and k < (j+1)*fold_length+window_size-half_observation_period:
                    continue
                if k >= i*fold_length and k < (i+1)*fold_length or k >= j*fold_length and k < end_index:
                    testIndices.append(k)
                else:
                    trainIndices.append(k)

            customCVindices.append((trainIndices, testIndices))

    return customCVindices

