import time
import random
import numpy as np
import pandas
from datetime import datetime
from keras_preprocessing import sequence
from torch.utils.data import Dataset
import argparse


# random seed setting
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# remove millisecond information of coarse
def time_process_coarse(dataset):
    for i in range(dataset.shape[0]):
        dataset[i, 4] = str(dataset[i, 4]).split('.')[0]
        dataset[i, 5] = str(dataset[i, 5]).split('.')[0]
    return dataset


# remove millisecond information of fine
def time_process_fine(dataset):
    for i in range(dataset.shape[0]):
        dataset[i, 5] = str(dataset[i, 5]).split('.')[0]
        dataset[i, 6] = str(dataset[i, 6]).split('.')[0]
    return dataset


# padding, since 0 is used as the padding value, in order to distinguish it from the label "0", here index+1
def find_value(data, value_list):
    index = 10000
    for i in range(len(value_list)):
        if data == value_list[i]:
            index = i
            break
    if index == 10000:
        print("cant find value")
    else:
        return index + 1


def load_dataset_factory_fine(file_path):
    print("loading datasets from :", file_path)
    dataframe = pandas.read_csv(file_path, header=0)
    dataframe = dataframe.fillna("null")
    column_names = dataframe.columns

    print("column list is : ")
    print(column_names)
    # Trace_ID,Sub_process_name,Activity_name,Reusable_devices,Non_reusable_devices,Start_time,End_time,Time_consuming,Worker_name,Station_number
    col_name_unique_dict = dict()
    col_name_index_dict = dict()

    data = dataframe.values
    for i in range(data.shape[1]):
        type_list = np.unique(data[:, i])
        col_name_unique_dict[column_names[i]] = type_list
        # padding 0 is also taken into account
        col_name_index_dict[column_names[i]] = np.arange(0, len(type_list) + 1)
    print(col_name_unique_dict)
    print(col_name_index_dict)

    # Time to remove milliseconds
    data = time_process_fine(data)

    # According to case id, divide training data and test data
    case_num = col_name_index_dict["Trace_ID"].shape
    #  train:test = 2:1
    clip_num = 3
    clip_length = int(case_num[0] / clip_num)
    datasetTR = data[data[:, 0] < 2 * clip_length]
    datasetTS = data[data[:, 0] >= 2 * clip_length]
    return generate_data_factory_fine(datasetTR, col_name_unique_dict), \
           generate_data_factory_fine(datasetTS, col_name_unique_dict), col_name_index_dict


def load_dataset_factory_coarse(file_path):
    print("loading datasets from :", file_path)
    dataframe = pandas.read_csv(file_path, header=0)
    dataframe = dataframe.fillna("null")
    column_names = dataframe.columns

    print("column list is : ")
    print(column_names)
    # Trace_ID,Sub_process_name,Sub_process_reusable_devices,Sub_process_non_reusable_devices,Start_time,End_time,Time_consuming,Worker_name,Station_number
    col_name_unique_dict = dict()
    col_name_index_dict = dict()

    data = dataframe.values
    for i in range(data.shape[1]):
        type_list = np.unique(data[:, i])
        col_name_unique_dict[column_names[i]] = type_list
        col_name_index_dict[column_names[i]] = np.arange(0, len(type_list) + 1)
    print(col_name_unique_dict)
    print(col_name_index_dict)

    data = time_process_coarse(data)

    case_num = col_name_index_dict["Trace_ID"].shape
    clip_num = 3
    clip_length = int(case_num[0] / clip_num)
    datasetTR = data[data[:, 0] < 2 * clip_length]
    datasetTS = data[data[:, 0] >= 2 * clip_length]
    return generate_data_factory_coarse(datasetTR, col_name_unique_dict), \
           generate_data_factory_coarse(datasetTS, col_name_unique_dict), col_name_index_dict


def load_dataset_BPI(file_path):
    dataframe = pandas.read_csv(file_path, header=0)
    dataframe = dataframe.fillna(0)
    column_names = dataframe.columns

    print("column list is : ")
    print(column_names)

    col_name_unique_dict = dict()
    col_name_index_dict = dict()

    data = dataframe.values
    for i in range(data.shape[1]):
        type_list = np.unique(data[:, i])
        col_name_unique_dict[column_names[i]] = type_list
        col_name_index_dict[column_names[i]] = np.arange(0, len(type_list) + 1)
    print(col_name_unique_dict)
    print(col_name_index_dict)

    case_num = col_name_index_dict["Case ID"].shape
    clip_num = 3
    clip_length = int(case_num[0] / clip_num)
    datasetTR = data[data[:, 0] < 2 * clip_length]
    datasetTS = data[data[:, 0] >= 2 * clip_length]
    return generate_data_bpi(datasetTR, col_name_unique_dict), \
           generate_data_bpi(datasetTS, col_name_unique_dict), col_name_index_dict


def load_dataset_helpdesk(file_path):
    dataframe = pandas.read_csv(file_path, header=0)
    dataframe = dataframe.fillna('null')
    column_names = dataframe.columns

    print("column list is : ")
    print(column_names)

    col_name_unique_dict = dict()
    col_name_index_dict = dict()

    data = dataframe.values
    for i in range(data.shape[1]):
        type_list = np.unique(data[:, i])
        col_name_unique_dict[column_names[i]] = type_list
        col_name_index_dict[column_names[i]] = np.arange(0, len(type_list) + 1)
    print(col_name_unique_dict)
    print(col_name_index_dict)

    # case_num = col_name_index_dict["CaseID"].shape
    # clip_num = 3
    # clip_length = int(case_num[0] / clip_num)
    # datasetTR = data[data[:, 0] < 2 * clip_length]
    # datasetTS = data[data[:, 0] >= 2 * clip_length]
    datasetTR = data[data[:, 0] < 199800]  # 2/3 cases
    datasetTS = data[data[:, 0] >= 199800]
    return generate_data_helpdesk(datasetTR, col_name_unique_dict), \
           generate_data_helpdesk(datasetTS, col_name_unique_dict), col_name_index_dict


def generate_data_factory_coarse(dataset, col_name_unique_dict):
    data = []
    newdataset = []
    temptarget = []
    # Trace_ID,Sub_process_name,Sub_process_reusable_devices,Sub_process_non_reusable_devices,Start_time,End_time,Time_consuming,Worker_name,Station_number
    caseID = dataset[0][0]

    Sub_process_name = dataset[0][1]
    Sub_process_name = find_value(Sub_process_name, col_name_unique_dict["Sub_process_name"])

    # Sub_process_reusable_devices = dataset[0][2]
    # Sub_process_reusable_devices = find_value(Sub_process_reusable_devices,
    #                                           col_name_unique_dict["Sub_process_reusable_devices"])

    # Sub_process_non_reusable_devices = dataset[0][3]
    # Sub_process_non_reusable_devices = find_value(Sub_process_non_reusable_devices,
    #                                               col_name_unique_dict["Sub_process_non_reusable_devices"])
    starttime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][5], "%Y-%m-%d %H:%M:%S")))
    lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][5], "%Y-%m-%d %H:%M:%S")))
    t = time.strptime(dataset[0][5], "%Y-%m-%d %H:%M:%S")
    Time_consuming = dataset[0][6]
    Worker_name = dataset[0][7]
    Worker_name = find_value(Worker_name, col_name_unique_dict["Worker_name"])
    Station_number = dataset[0][8]
    Station_number = find_value(Station_number, col_name_unique_dict["Station_number"])
    n = 1  # Worker counter for the current case
    # midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    # timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
    temptarget.append(datetime.fromtimestamp(time.mktime(t)))

    a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
    a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
    # a.append(timesincemidnight)
    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
    a.append(Sub_process_name)
    # a.append(Sub_process_reusable_devices)
    # a.append(Sub_process_non_reusable_devices)
    a.append(Time_consuming)
    a.append(Worker_name)
    # a.append(Sub_process_name)
    a.append(Station_number)
    newdataset.append(a)

    for line in dataset[1:, :]:
        # print line
        case = line[0]
        Sub_process_name = line[1]
        Sub_process_name = find_value(Sub_process_name, col_name_unique_dict["Sub_process_name"])
        # Sub_process_reusable_devices = line[2]
        # Sub_process_reusable_devices = find_value(Sub_process_reusable_devices,
        #                                           col_name_unique_dict["Sub_process_reusable_devices"])
        # Sub_process_non_reusable_devices = line[3]
        # Sub_process_non_reusable_devices = find_value(Sub_process_non_reusable_devices,
        #                                               col_name_unique_dict["Sub_process_non_reusable_devices"])
        Time_consuming = line[6]
        # Time_consuming = find_value(Time_consuming, col_name_unique_dict["Time_consuming"])
        Worker_name = line[7]
        Worker_name = find_value(Worker_name, col_name_unique_dict["Worker_name"])
        Station_number = line[8]
        Station_number = find_value(Station_number, col_name_unique_dict["Station_number"])

        if case == caseID:
            t = time.strptime(line[5], "%Y-%m-%d %H:%M:%S")
            # midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            # timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            # a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(Sub_process_name)
            # a.append(Sub_process_reusable_devices)
            # a.append(Sub_process_non_reusable_devices)
            a.append(Time_consuming)
            a.append(Worker_name)
            # a.append(Sub_process_name)
            a.append(Station_number)
            newdataset.append(a)
            lastevtime = datetime.fromtimestamp(time.mktime(t))
            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            n += 1
            finishtime = datetime.fromtimestamp(time.mktime(t))

        else:

            caseID = case
            for i in range(1, len(newdataset)):  # +1 not adding last case. target is 0, not interesting. era 1
                data.append(newdataset[:i])

            newdataset = []

            starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[5], "%Y-%m-%d %H:%M:%S")))
            lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(line[5], "%Y-%m-%d %H:%M:%S")))
            t = time.strptime(line[5], "%Y-%m-%d %H:%M:%S")
            # midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            # timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            # a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(Sub_process_name)
            # a.append(Sub_process_reusable_devices)
            # a.append(Sub_process_non_reusable_devices)
            a.append(Time_consuming)
            a.append(Worker_name)
            # a.append(Sub_process_name)
            a.append(Station_number)
            newdataset.append(a)

            for i in range(n):
                temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            temptarget.pop()  # remove last element with zero target

            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            finishtime = datetime.fromtimestamp(time.mktime(t))
            n = 1

    for i in range(1, len(newdataset)):  # + 1 not adding last event, target is 0 in that case. era 1
        data.append(newdataset[:i])
    for i in range(n):
        temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
    temptarget.pop()  # remove last element with zero target

    print("Generated dataset with n_samples:", len(temptarget))
    assert (len(temptarget) == len(data))
    return data, temptarget


def generate_data_factory_fine(dataset, col_name_unique_dict):
    data = []
    newdataset = []
    temptarget = []
    # Trace_ID,Sub_process_name,Activity_name,Reusable_devices,Non_reusable_devices,Start_time,End_time,Time_consuming,Worker_name,Station_number
    # Process the first row of data

    # case ID of the first event of the first case
    caseID = dataset[0][0]

    # Sub_process_name of the first event of the first case
    # Sub_process_name = dataset[0][1]
    # Sub_process_name = find_value(Sub_process_name, col_name_unique_dict["Sub_process_name"])

    # Activity name of the first event of the first case
    Activity_name = dataset[0][2]
    Activity_name = find_value(Activity_name, col_name_unique_dict["Activity_name"])

    # Sub_process_reusable_devices of the first event of the first case
    # Reusable_devices = dataset[0][3]
    # Reusable_devices = find_value(Reusable_devices, col_name_unique_dict["Reusable_devices"])

    # Sub_process_non_reusable_devices of the first event of the first case
    # Non_reusable_devices = dataset[0][4]
    # Non_reusable_devices = find_value(Non_reusable_devices, col_name_unique_dict["Non_reusable_devices"])

    # For unified processing with other data set time information, use End_time as the start time, end time,
    # and current time starttime, lasttime respectively represent the start time and end time of the current case
    starttime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][6], "%Y-%m-%d %H:%M:%S")))
    lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][6], "%Y-%m-%d %H:%M:%S")))
    t = time.strptime(dataset[0][6], "%Y-%m-%d %H:%M:%S")

    # Time_consuming of the first event of the first case
    Time_consuming = dataset[0][7]
    # Time_consuming = find_value(Time_consuming, col_name_unique_dict["Time_consuming"])

    # Worker_name of the first event of the first case
    Worker_name = dataset[0][8]
    Worker_name = find_value(Worker_name, col_name_unique_dict["Worker_name"])

    # Station_number of the first event of the first case
    Station_number = dataset[0][9]
    Station_number = find_value(Station_number, col_name_unique_dict["Station_number"])

    # Midnight time at the current time of the current event
    # midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    # The time interval between the current time of the current event and the midnight time where it is located
    # timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()

    n = 1  # Worker counter for the current case
    temptarget.append(datetime.fromtimestamp(time.mktime(t)))

    # save information-----------------------------------------------------------------------------------------------------------
    # Save the timestamp of the current time and the case start time, information-1
    a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
    # Save the timestamp of the current time and the case end time, information-2
    a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
    # Save the current time and the midnight timestamp, information-3
    # a.append(timesincemidnight)
    # Save the current time as the day of the week, information-4
    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
    # save Sub_process_name, information-5
    # a.append(Sub_process_name)
    # save Activity_name, save -6
    a.append(Activity_name)
    # save Sub_process_reusable_devices, information-7
    # a.append(Reusable_devices)
    # save Sub_process_non_reusable_devices, information-8
    # a.append(Non_reusable_devices)
    # save Time_consuming, information-9
    a.append(Time_consuming)
    # save Worker_name, information-10
    a.append(Worker_name)
    # a.append(Activity_name)
    # save Station_number, information-11
    a.append(Station_number)
    newdataset.append(a)

    # Process other rows of data
    for line in dataset[1:, :]:
        case = line[0]
        # Sub_process_name = line[1]
        # Sub_process_name = find_value(Sub_process_name, col_name_unique_dict["Sub_process_name"])
        Activity_name = line[2]
        Activity_name = find_value(Activity_name, col_name_unique_dict["Activity_name"])
        # Reusable_devices = line[3]
        # Reusable_devices = find_value(Reusable_devices, col_name_unique_dict["Reusable_devices"])
        # Non_reusable_devices = line[4]
        # Non_reusable_devices = find_value(Non_reusable_devices, col_name_unique_dict["Non_reusable_devices"])
        Time_consuming = line[7]
        Worker_name = line[8]
        Worker_name = find_value(Worker_name, col_name_unique_dict["Worker_name"])
        Station_number = line[9]
        Station_number = find_value(Station_number, col_name_unique_dict["Station_number"])

        # If the data and the previous data belong to the same case, that is, the event of the same case
        if case == caseID:
            t = time.strptime(line[6], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            # a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            # a.append(Sub_process_name)
            a.append(Activity_name)
            # a.append(Reusable_devices)
            # a.append(Non_reusable_devices)
            a.append(Time_consuming)
            a.append(Worker_name)
            # a.append(Activity_name)
            a.append(Station_number)
            newdataset.append(a)
            # Update the lastevtime of the current case with the current time
            lastevtime = datetime.fromtimestamp(time.mktime(t))
            # Except for the first row of data, temporarily use the current time as the remaining time label
            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            n += 1  # sub_process num counter of current case +1
            # Update the final time of the case with the current time
            finishtime = datetime.fromtimestamp(time.mktime(t))

        else:
            # When switching to another case
            caseID = case
            #
            for i in range(1, len(newdataset)):  # +1 not adding last case. target is 0, not interesting. era 1
                data.append(newdataset[:i])

            newdataset = []
            # For a new case, use the End_time of the first event of the new case as the start time, end time, and current time
            # Process the first event of the new case in the original way, and store

            starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[6], "%Y-%m-%d %H:%M:%S")))
            lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(line[6], "%Y-%m-%d %H:%M:%S")))
            t = time.strptime(line[6], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            # a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            # a.append(Sub_process_name)
            a.append(Activity_name)
            # a.append(Reusable_devices)
            # a.append(Non_reusable_devices)
            a.append(Time_consuming)
            a.append(Worker_name)
            # a.append(Activity_name)
            a.append(Station_number)
            newdataset.append(a)

            # The previous case has ended
            # How many sub_processes in the previous case
            for i in range(n):
                # Process the current time in temptarget as the remaining time in seconds
                temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            # temptarget stores the remaining time of each event in the previous case
            # The remaining time of the last event of the previous case is 0, remove it
            temptarget.pop()  # remove last element with zero target

            # Update the new remaining time label of the new case and the finishtime and number of events
            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            finishtime = datetime.fromtimestamp(time.mktime(t))
            n = 1
            # In the end, the remaining time of all cases will be stored in temptarget, and other information prefix information will be stored in data, neither of which contains the last event of each case

    # Process the last case. Since each case data is budgeted for the remaining time from the beginning of the next case, the last case needs to be calculated separately.
    for i in range(1, len(newdataset)):  # + 1 not adding last event, target is 0 in that case. era 1
        data.append(newdataset[:i])
    for i in range(n):
        temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
    temptarget.pop()  # remove last element with zero target

    print("Generated dataset with n_samples:", len(temptarget))
    assert (len(temptarget) == len(data))
    return data, temptarget


def generate_data_bpi(dataset, col_name_unique_dict):
    data = []
    newdataset = []
    temptarget = []

    caseID = dataset[0][0]
    Activity = dataset[0][1]
    Activity = find_value(Activity, col_name_unique_dict["Activity"])
    starttime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
    lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
    t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
    Resource = dataset[0][3]
    AMOUNT_REQ = dataset[0][4]
    concept_name = dataset[0][5]
    concept_name = find_value(concept_name, col_name_unique_dict["concept:name"])
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()  # 当前时间与午夜时间的秒数间隔
    n = 1
    temptarget.append(datetime.fromtimestamp(time.mktime(t)))

    a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
    a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
    a.append(timesincemidnight)
    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
    a.append(Activity)
    a.append(Resource)
    a.append(AMOUNT_REQ)
    a.append(concept_name)
    newdataset.append(a)

    for line in dataset[1:, :]:
        # print line
        case = line[0]

        Activity = line[1]
        Activity = find_value(Activity, col_name_unique_dict["Activity"])
        Resource = line[3]
        AMOUNT_REQ = line[4]
        concept_name = line[5]
        concept_name = find_value(concept_name, col_name_unique_dict["concept:name"])

        if case == caseID:
            t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(Activity)
            a.append(Resource)
            a.append(AMOUNT_REQ)
            a.append(concept_name)
            newdataset.append(a)
            lastevtime = datetime.fromtimestamp(time.mktime(t))
            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            n += 1
            finishtime = datetime.fromtimestamp(time.mktime(t))

        else:
            caseID = case
            for i in range(1, len(newdataset)):  # +1 not adding last case. target is 0, not interesting. era 1
                data.append(newdataset[:i])

            newdataset = []

            starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(Activity)
            a.append(Resource)
            a.append(AMOUNT_REQ)
            a.append(concept_name)
            newdataset.append(a)

            for i in range(n):
                temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            temptarget.pop()  # remove last element with zero target

            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            finishtime = datetime.fromtimestamp(time.mktime(t))
            n = 1

    for i in range(1, len(newdataset)):  # + 1 not adding last event, target is 0 in that case. era 1
        data.append(newdataset[:i])
    for i in range(n):
        temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
    temptarget.pop()  # remove last element with zero target

    print("Generated dataset with n_samples:", len(temptarget))
    assert (len(temptarget) == len(data))
    return data, temptarget


def generate_data_helpdesk(dataset, col_name_unique_dict):
    data = []
    newdataset = []
    temptarget = []

    caseID = dataset[0][0]
    ActivityID = dataset[0][1]
    ActivityID = find_value(ActivityID, col_name_unique_dict["ActivityID"])

    starttime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
    lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
    t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()  # 当前时间与午夜时间的秒数间隔
    n = 1
    temptarget.append(datetime.fromtimestamp(time.mktime(t)))

    a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
    a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
    a.append(timesincemidnight)
    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
    a.append(ActivityID)
    newdataset.append(a)

    for line in dataset[1:, :]:
        # print line
        case = line[0]
        ActivityID = line[1]
        ActivityID = find_value(ActivityID, col_name_unique_dict["ActivityID"])
        if case == caseID:
            t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(ActivityID)
            newdataset.append(a)
            lastevtime = datetime.fromtimestamp(time.mktime(t))
            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            n += 1
            finishtime = datetime.fromtimestamp(time.mktime(t))

        else:
            caseID = case
            for i in range(1, len(newdataset)):  # +1 not adding last case. target is 0, not interesting. era 1
                data.append(newdataset[:i])

            newdataset = []

            starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            lastevtime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            a = [(datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(time.mktime(t)) - lastevtime).total_seconds())
            a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.append(ActivityID)
            newdataset.append(a)

            for i in range(n):
                temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()

            temptarget.pop()  # remove last element with zero target

            temptarget.append(datetime.fromtimestamp(time.mktime(t)))
            finishtime = datetime.fromtimestamp(time.mktime(t))
            n = 1

    for i in range(1, len(newdataset)):  # + 1 not adding last event, target is 0 in that case. era 1
        data.append(newdataset[:i])
    for i in range(n):
        temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
    temptarget.pop()  # remove last element with zero target

    print("Generated dataset with n_samples:", len(temptarget))
    assert (len(temptarget) == len(data))
    return data, temptarget


def dataset_path(dataset_name):
    if dataset_name == "factory_coarse":
        # return "../log_simulation/sub_process_level_log_all.csv"
        return "../data/Process_level_log_all.csv"
    elif dataset_name == "factory_fine":
        # return "../log_simulation/Activity_level_log_all.csv"
        return "../data/Activity_level_log_all.csv"
    elif dataset_name == "BPIC-12":
        return "../BPI_data/BPI_12_anonimyzed.csv"
    elif dataset_name == "helpdesk":
        # return "../BPI_data/helpdesk_extend.csv"
        return "../BPI_data/BPIC_2012_A.csv"
    else:
        print("wrong dataset name")
        return "Wrong"


class TFE_Dataset(Dataset):
    def __init__(self, X, Y):
        # TODO
        # 1. Initialize file path or list of file names.
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return self.X[index], self.Y[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='seed', type=int, default=20, required=False)
    # parser.add_argument('--dataset_name', help='dataset name', type=str, default="factory_coarse", required=False)
    # parser.add_argument('--dataset_name', help='dataset name', type=str, default="BPIC-12", required=False)
    parser.add_argument('--dataset_name', help='dataset name', type=str, default="factory_fine", required=False)
    # parser.add_argument('--dataset_name', help='dataset name', type=str, default="helpdesk", required=False)
    args = parser.parse_args()
    setup_seed(args.seed)
    dataset_file_path = dataset_path(dataset_name=args.dataset_name)
    if args.dataset_name == "factory_coarse":
        # 51246, 13, 6
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_factory_coarse(dataset_file_path)
    elif args.dataset_name == "BPIC-12":
        # 42748, 73, 8
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_BPI(dataset_file_path)
    elif args.dataset_name == "factory_fine":
        # 62000, 31, 6
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_factory_fine(dataset_file_path)
    elif args.dataset_name == "helpdesk":
        # 5481, 13, 5
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_helpdesk(dataset_file_path)

    X_train = sequence.pad_sequences(X_train)
    print("DEBUG: training shape", X_train.shape)
    print("")
