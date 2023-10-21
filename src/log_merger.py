# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/15 10:00
@Auther ： Zzou
@File ：log_merger.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ： merge logs as one file
"""

import pandas as pd


def log_merge(mode, log_file_num, csv_file):
    # Attributes storage
    trace_id = list()
    sp_name = list()
    an = list()
    Rd = list()
    NRd = list()
    s_time = list()
    e_time = list()
    atv = list()
    wn = list()
    Sn = list()
    if mode == "fine":
        for i in range(log_file_num):
            index = i + 1
            file_path = "../data/Activity_level_log_" + str(index) + ".csv"
            data = pd.read_csv(file_path, encoding='utf_8_sig')
            ti = data["Trace_ID"].values + i * 300
            trace_id.extend(ti)
            sp_name.extend(data["Sub_process_name"].values)
            an.extend(data["Activity_name"].values)
            Rd.extend(data["Reusable_devices"].values)
            NRd.extend(data["Non_reusable_devices"].values)
            s_time.extend(data["Start_time"].values)
            e_time.extend(data["End_time"].values)
            atv.extend(data["Time_consuming"].values)
            wn.extend(data["Worker_name"].values)
            Sn.extend(data["Station_number"].values)
        # convert data into csv with dataframe
        dataframe = pd.DataFrame(
            {'Trace_ID': trace_id, 'Sub_process_name': sp_name, 'Activity_name': an, 'Reusable_devices': Rd,
             'Non_reusable_devices': NRd,
             'Start_time': s_time, 'End_time': e_time, 'Time_consuming': atv, 'Worker_name': wn, 'Station_number': Sn})
        dataframe.to_csv(csv_file, index=False, sep=',', encoding='utf_8_sig')

    elif mode == "coarse":
        for i in range(log_file_num):
            index = i + 1
            file_path = "../data/Process_level_log_" + str(index) + ".csv"
            data = pd.read_csv(file_path, encoding='utf_8_sig')
            ti = data["Trace_ID"].values + i * 300
            trace_id.extend(ti)
            sp_name.extend(data["Sub_process_name"].values)
            Rd.extend(data["Sub_process_reusable_devices"].values)
            NRd.extend(data["Sub_process_non_reusable_devices"].values)
            s_time.extend(data["Start_time"].values)
            e_time.extend(data["End_time"].values)
            atv.extend(data["Time_consuming"].values)
            wn.extend(data["Worker_name"].values)
            Sn.extend(data["Station_number"].values)
        # convert data into csv with dataframe
        dataframe = pd.DataFrame(
            {'Trace_ID': trace_id, 'Sub_process_name': sp_name, 'Sub_process_reusable_devices': Rd,
             'Sub_process_non_reusable_devices': NRd,
             'Start_time': s_time, 'End_time': e_time, 'Time_consuming': atv, 'Worker_name': wn, 'Station_number': Sn})
        dataframe.to_csv(csv_file, index=False, sep=',', encoding='utf_8_sig')


if __name__ == "__main__":
    Mode = "coarse"
    Log_file_num = 20
    Csv_file = "../data/Process_level_log_all.csv"
    assert Mode == "fine" or Mode == "coarse", "Wrong log granularity mode!"
    log_merge(mode=Mode, log_file_num=Log_file_num, csv_file=Csv_file)
