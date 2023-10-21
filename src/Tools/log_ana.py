# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/11 9:58
@Auther ： Zzou
@File ：log_ana.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""


def get_dataset_path(dataset_name):
    if dataset_name == "factory_coarse":
        return "../../data/Process_level_log_all.csv"
    elif dataset_name == "factory_fine":
        return "../../data/Activity_level_log_all.csv"
    elif dataset_name == "BPIC-12":
        return "../BPI_data/BPI_12_anonimyzed.csv"
    elif dataset_name == "helpdesk":
        return "../BPI_data/helpdesk_extend.csv"
    else:
        print("wrong dataset name")
        return "Wrong"
