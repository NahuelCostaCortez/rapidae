"""
Class that contains the different preprocessors that can be used to preprocess the data.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(
        name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - \
        result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
        df_op_cond['setting_2'].astype(str) + '_' + \
        df_op_cond['setting_3'].astype(str)

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']
                   == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
        
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr', group_keys=False)[sensors].apply(
        lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit_nr')['unit_nr'].transform(
        create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    data_gen = (list(gen_train_data(df[df['unit_nr'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)

    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    label_gen = [gen_labels(df[df['unit_nr'] == unit_nr], sequence_length, label)
                 for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)

    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(
            columns)), fill_value=mask_value)  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def CMAPSS_preprocessor(train, test, y_test, sensors=['s_3', 's_4', 's_7', 's_11', 's_12'], sequence_length=30, alpha=0.1, threshold=125):
    '''
    Preprocesses the CMAPSS data and yields train, test and validation data/labels.

    Args
    ----
        train (DataFrame): Training data.
        test (DataFrame): Test data.
        y_test (DataFrame): Test labels.
        sensors (list): List of sensors to use.
        sequence_length (int): Length of the sequence.
        alpha (float): Smoothing parameter.
        threshold (int): Threshold for RUL values.

    Returns
    -------
        x_train, y_train, x_val, y_val, x_test, y_test preprocessed
    '''
    # create RUL values according to the piece-wise target function

    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    # remove unused sensors
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    drop_sensors = [
        element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(
        X_train_pre, X_test_pre, sensors)

    # exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        # gss returns indexes and index starts at 1
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

    x_train = gen_data_wrapper(
        X_train_pre, sequence_length, sensors, train_unit)
    y_train = gen_label_wrapper(
        X_train_pre, sequence_length, ['RUL'], train_unit)

    x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
    y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

    # create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']

