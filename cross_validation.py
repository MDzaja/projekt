import enum

class CV(enum.Enum):
   train = 1
   test = 2
   skip = 3

#create a list of data marked for cross validation, each element has 8 train folds and 2 test folds
#after each test fold there is a skip period where no data is used
def ten_fold_cv_selection(data, observe_prev_N_days):
    selection_list = []
    fold_length = int(len(data)/10)
    half_observation_period = int(observe_prev_N_days/2)
    for i in range(0, 10):
        for j in range(i+1, 10):
            data_copy = data.copy()

            data_copy['CV'] = CV.train
            data_copy.iloc[i*fold_length:(i+1)*fold_length, data_copy.columns.get_loc('CV')] = CV.test
            end_index = (j+1)*fold_length
            if j == 9:
                end_index = len(data_copy)
            data_copy.iloc[j*fold_length:end_index, data_copy.columns.get_loc('CV')] = CV.test
            if j-i != 1:
                data_copy.iloc[(i+1)*fold_length-half_observation_period: \
                    (i+1)*fold_length+observe_prev_N_days-half_observation_period, \
                    data_copy.columns.get_loc('CV')] = CV.skip
            if j != 9:
                data_copy.iloc[(j+1)*fold_length-half_observation_period: \
                    (j+1)*fold_length+observe_prev_N_days-half_observation_period, \
                    data_copy.columns.get_loc('CV')] = CV.skip

            selection_list.append(data_copy)

    return selection_list

    