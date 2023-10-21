# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/15 10:00
@Auther ： Zzou
@File ：order_rules.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：Determine process sequence set
"""

import itertools
import pandas as pd
import random


# Mode 1: Simulate the process sequence relationship when the station sequence is variable
# (variable sequence pipeline log simulation, from the perspective of process evaluation)
def sp_generation_wcsnsp(snsp):
    # According to the sequence relationship.vsdx, cut the sequence fragment
    # Generate all subprocess sequences, before 9
    sp_list1 = [13, 245, 6, 7, 8]
    sp_list_index1 = dict()
    tmp_list1 = itertools.permutations(sp_list1)
    res_list1 = list()
    for one in tmp_list1:
        res_list1.append(list(one))
    # Filter sequences that match the before-and-after rules
    new_res_list1 = list()
    for order in res_list1:
        for i, sp in enumerate(order):
            sp_list_index1[sp] = i
        if sp_list_index1[13] < sp_list_index1[7] and sp_list_index1[245] < sp_list_index1[8]:
            new_res_list1.append(order)
    #  Revert to process granularity sequence
    res_list1 = list()
    for i in range(len(new_res_list1)):
        order = new_res_list1[i]
        order_temp1 = list()
        order_temp2 = list()
        for item in order:
            if item == 13:
                order_temp1.append(1)
                order_temp1.append(3)
                order_temp2.append(3)
                order_temp2.append(1)
            elif item == 245:
                order_temp1.append(2)
                order_temp1.append(4)
                order_temp1.append(5)
                order_temp2.append(4)
                order_temp2.append(2)
                order_temp2.append(5)
            else:
                order_temp1.append(item)
                order_temp2.append(item)
        res_list1.append(order_temp1)
        res_list1.append(order_temp2)
    # Generate all process sequences, after 9
    res_list2 = [[9, 10, 11, 12, 13, 14], [9, 10, 11, 13, 12, 14]]
    results_order = list()
    for item in res_list1:
        results_order.append(item + res_list2[0])
        results_order.append(item + res_list2[1])
    # convert list to string
    results_order_string = list()
    for order in results_order:
        str_ = ""
        for sp in order:
            str_ += ","
            str_ += str(sp)
        str_ = str_[1:]
        results_order_string.append(str_)

    # Process interval (transmission distance of conveyor belt)
    results_interval_string = list()
    for order in results_order:
        str_ = "0"
        for i in range(len(order) - 1):
            str_ += ","
            if snsp[str(order[i])] != snsp[str(order[i + 1])]:
                str_ += "3.0"  # Conveyor transport between different stations for 3 seconds
            elif snsp[str(order[i])] == snsp[str(order[i + 1])]:
                str_ += "0.5"  # The same station, there is a 0.5 second interval
        results_interval_string.append(str_)
    return results_order_string, results_interval_string


# Mode 2, the process sequence relationship is simulated when the station sequence is fixed, and it can be filtered from all cases
def sp_generation_wocsnsp(results_order_string, results_interval_string, template, snsp):
    results_order_string_2 = list()
    results_interval_string_2 = list()
    for i in range(len(results_order_string)):
        order = results_order_string[i].split(",")
        temp_set = list()
        for sp in order:
            sn = snsp[str(sp)]
            temp_set.append(sn)
        temp_set_ = list(set(temp_set))
        temp_set_.sort(key=temp_set.index)
        if temp_set_ == template:
            results_interval_string_2.append(results_interval_string[i])
            results_order_string_2.append(results_order_string[i])
    return results_order_string_2, results_interval_string_2


def process_bundling():
    #  All process sequences in the case of process bundling
    l1 = [[1, 2, 6], [1, 6, 2], [2, 1, 6], [2, 6, 1], [6, 1, 2], [6, 2, 1]]
    l2 = [[9, 10, 11, 12, 13, 14], [9, 10, 11, 13, 12, 14]]
    result = list()
    for item in l1:
        result.append(item + l2[0])
        result.append(item + l2[1])
    print(result)
    num = 100
    data = list()
    ID = list()
    Sp = list()
    for i in range(num):
        random_index = random.randint(0, len(result) - 1)
        data.append(result[random_index])
    print(data)
    for i, d in enumerate(data):
        for item in d:
            ID.append(i)
            Sp.append(item)
    dataframe = pd.DataFrame({'Trace_ID': ID, 'Sub_process_name': Sp})
    dataframe.to_csv("bundle_sub_process_level_log.csv", index=False, sep=',', encoding='utf_8_sig')


if __name__ == "__main__":
    # Correspondence between process and station
    snsp = dict()
    snsp['1'] = '001'
    snsp['2'] = '002'
    snsp['3'] = '001'
    snsp['4'] = '002'
    snsp['5'] = '003'
    snsp['6'] = '004'
    snsp['7'] = '005'
    snsp['8'] = '006'
    snsp['9'] = '007'
    snsp['10'] = '008'
    snsp['11'] = '009'
    snsp['12'] = '010'
    snsp['13'] = '011'
    snsp['14'] = '012'
    # sequence of processes
    orsp = ['1,7', '2,8', '3,9', '4,9', '5,9', '4,5', '6,9', '7,9', '8,9', '9,10', '10,11', '11,12', '11,13', '12,14',
            '13,14']
    # Collection of all process sequences
    results_order_string, results_interval_string = sp_generation_wcsnsp(snsp=snsp)
    # Process sequences when station sequence(template) is specified
    template = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
    Station_interval = ['0', '3.0', '3.2', '3.3', '3.0', '2.8', '3.0', '3.2', '2.7', '2.8', '3.1', '3.3']
    results_order_string_fixed, results_interval_string_fixed = sp_generation_wocsnsp(results_order_string,
                                                                                      results_interval_string, template,
                                                                                      snsp)

    # In fact, the distance of the station conveyor belt is different
    # So rewrite the results_interval_string_fixed as results_interval_string_fixed_new
    results_interval_string_fixed_new = []
    for item in results_order_string_fixed:
        item = item.split(",")
        interval = "0"
        pre = item[0]
        for i in range(len(item) - 1):
            p = item[i + 1]
            if snsp[p] == snsp[pre]:
                interval += ",0.5"
            elif snsp[p] != snsp[pre]:
                station = snsp[p]
                index = template.index(station)
                interval = interval + "," + Station_interval[index]
            pre = p
        results_interval_string_fixed_new.append(interval)

    print(results_order_string_fixed)
    print(results_interval_string_fixed)
    print(results_interval_string_fixed_new)
