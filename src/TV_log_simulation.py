# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/15 11:19
@Auther ： Zzou
@File ：TV_log_simulation.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import datetime
import time
import numpy
import random
from order_rules import *
import pandas as pd
import csv

# task category
assemblage_label = dict()
assemblage_label['1'] = "将按键板卡装到按键支架上"
assemblage_label['2'] = "将遥控板黏贴到遥控导光柱上"
assemblage_label['3'] = "插装按键-遥控线"
assemblage_label['4'] = "将外接电源端子的线插装在电源开关上"
assemblage_label['5'] = "将开关-电源板连接线插装到电源开关上"
assemblage_label['6'] = "首先确认工装衬垫上无异物避免划伤液晶屏,如有异物及灰尘请用毛刷清理干净"
assemblage_label['7'] = "小心拿出液晶屏,不要揭掉液晶屏幕保护膜和侧边保护膜,检验液晶屏是否存在划伤等不良"
assemblage_label['8'] = "将屏放在工装板上"
assemblage_label['9'] = "插装LVDS线并使用一条胶带固定"
assemblage_label['10'] = "将端子板卡装到主板上"
assemblage_label['11'] = "将主板组件放置到液晶屏上"
assemblage_label['12'] = "使用螺钉固定主板和端子板"
assemblage_label['13'] = "将扬声器组件卡装到背板上并使用螺钉固定"
assemblage_label['14'] = "将按键组件、遥控组件卡装到图示位置"
assemblage_label['15'] = "本工位执行MES采集工作,将2联条码标签黏贴在液晶屏上,注意其中一联保留不干胶"
assemblage_label['16'] = "将液晶屏背面的BIN值记录到条码标签背面,便于完检更改BIN值,注意是否更改以完检要求为准"
assemblage_label['17'] = "将电源开关端子支架固定到背板上"
assemblage_label['18'] = "电源开关组件卡装,注意方向开关"
assemblage_label['19'] = "按图示将各导线插接"
assemblage_label['20'] = "按图示使用耐高温胶带将各连接线固定到背板上"
assemblage_label['21'] = "对机内进行检验"
assemblage_label['22'] = "在后壳端子板处两凹槽内粘贴左侧端子标牌和右侧端子标牌"
assemblage_label['23'] = "在后壳居中处凹槽内粘贴铭牌"
assemblage_label['24'] = "按图示使用六颗螺钉固定后壳(上半区)"
assemblage_label['25'] = "按图示安装六颗螺钉固定后壳(下半区)"
assemblage_label['26'] = "整机完成电性能检验后,拔掉电源线"
assemblage_label['27'] = "将1联条码标签粘贴在后壳后视右下角位置"
assemblage_label['28'] = "检查后盖所有螺钉固定到位"
assemblage_label['29'] = "检查标签粘贴是否牢固"
assemblage_label['30'] = "检查整机外观无划伤、碰伤、掉漆等不正常现象"
assemblage_label['31'] = "擦净电视机各部位"
assemblage_label['32'] = "检查前后壳闪缝、屏壳闪缝符合企业标准"

# Worker-Process Reference Table
assemblage_Station_worker_name = dict()
assemblage_Station_worker_name['001'] = ["Mary", "Alex", "Linda"]
assemblage_Station_worker_name['002'] = ["Linda", "Bert", "Mary"]
assemblage_Station_worker_name['003'] = ["John", "Cary", "Cary"]
assemblage_Station_worker_name['004'] = ["Robert", "Daniel", "Robert"]
assemblage_Station_worker_name['005'] = ["Peter", "Evan", "Steven"]
assemblage_Station_worker_name['006'] = ["Helen", "Todd", "Mike"]
assemblage_Station_worker_name['007'] = ["Steven", "Kevin", "Helen"]
assemblage_Station_worker_name['008'] = ["Frank", "Leonard", "Leonard"]
assemblage_Station_worker_name['009'] = ["Jack", "Mike", "Evan"]
assemblage_Station_worker_name['010'] = ["Tony", "Ray", "Ray"]
assemblage_Station_worker_name['011'] = ["Lee", "Robin", "Robin"]
assemblage_Station_worker_name['012'] = ["Tom", "Thomas", "Tom"]

# Work ability attributes of workers, the larger the value, the stronger the ability
# Perhaps work ability should be linked to specific processes
worker_work_efficiency = dict()
worker_work_efficiency["Mary"] = 1.0
worker_work_efficiency["Alex"] = 0.9
worker_work_efficiency["Linda"] = 1.1
worker_work_efficiency["Bert"] = 1.0
worker_work_efficiency["John"] = 1.2
worker_work_efficiency["Cary"] = 1.0
worker_work_efficiency["Robert"] = 0.7
worker_work_efficiency["Daniel"] = 1.0
worker_work_efficiency["Peter"] = 0.8
worker_work_efficiency["Evan"] = 1.0
worker_work_efficiency["Helen"] = 1.0
worker_work_efficiency["Todd"] = 1.0
worker_work_efficiency["Steven"] = 0.7
worker_work_efficiency["Kevin"] = 1.0
worker_work_efficiency["Frank"] = 1.0
worker_work_efficiency["Leonard"] = 1.0
worker_work_efficiency["Jack"] = 1.0
worker_work_efficiency["Mike"] = 1.0
worker_work_efficiency["Tony"] = 0.9
worker_work_efficiency["Ray"] = 0.8
worker_work_efficiency["Lee"] = 1.0
worker_work_efficiency["Robin"] = 1.1
worker_work_efficiency["Tom"] = 1.2
worker_work_efficiency["Thomas"] = 1.0

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

# Generate all possible sequence of operations
# Unlimited work order
results_orders_string, results_intervals_string = sp_generation_wcsnsp(snsp=snsp)
# Limit station sequence
template = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
results_order_fixed_string, results_interval_fixed_string = sp_generation_wocsnsp(results_orders_string,
                                                                                  results_intervals_string, template,
                                                                                  snsp)

# # save all traces of unlimited work order
# data = []
# for i in range(len(results_orders_string)):
#     trace_id = i
#     events = results_orders_string[i].split(',')
#     # Check the pipeline for an incomplete set of steps
#     if len(events) != 14:
#         print("-----------------------------")
#     else:
#         data.append([trace_id, '0'])
#         for j in range(len(events)):
#             data.append([trace_id, events[j]])
# with open('trace-0.csv', 'w', newline='') as f:
#     cw = csv.writer(f, dialect='excel')
#     cw.writerow(['CaseID', 'Event'])
#     for item in data:
#         cw.writerow(item)

# Process execution information dictionary
assemblage_name_dict = dict()
assemblage_name_dict['Sub-process orders'] = [results_order_fixed_string[0]]
assemblage_name_dict['Sub-process intervals'] = [results_interval_fixed_string[0]]

# Process execution information
assemblage_name_dict['1'] = dict()
assemblage_name_dict['1']['name'] = "按键组件"  # process name
assemblage_name_dict['1']['asm'] = "1,2,3"  # The steps included in the process
assemblage_name_dict['1'][
    'Reusable devices'] = "防静电腕带,涤锦符合切指手套"  # A list of the number of reusable resources in a process, the default
# number is 1, and other numbers are marked with *num
assemblage_name_dict['1'][
    'Non-reusable devices'] = "按键组件,按键板组件,导光件,遥控板组件,条形连接器"  # The list of non-reusable resources (materials) in the
# process, the default quantity is 1, and other quantities are marked with *num
# Resources are grouped at the step level
assemblage_name_dict['1']['asm-Reusable devices'] = ["防静电腕带,涤锦符合切指手套", "防静电腕带,涤锦符合切指手套",
                                                     "防静电腕带,涤锦符合切指手套"]  # A list of the number of reusable resources
# for each process, the default number is 1, and other numbers are marked with *num
# non-reusable resources are grouped at the step level
assemblage_name_dict['1']['asm-Non-reusable devices'] = ["按键组件,按键板组件", "导光件,遥控板组件",
                                                         "条形连接器"]  # List of non-reusable resources (materials) for
# each process, the default quantity is 1, and other quantities are marked with *num
assemblage_name_dict['1']['total time'] = "27.13"  # The average completion time of the process
assemblage_name_dict['1'][
    'total time variance'] = "2"  # The fluctuation range of the process completion time (total time +- variance)
assemblage_name_dict['1']['Station number'] = "001"  # Process station number
assemblage_name_dict['1']['order list'] = ["1,2,3", "2,1,3"]  # Process step execution list
assemblage_name_dict['1']['time list'] = ["15.0,5.0,7.1", "5.2,15.0,6.8"]  # Process step evg execution time list
assemblage_name_dict['1']['time variance list'] = ["1.1,0.5,0.6",
                                                   "0.5,1.1,0.6"]  # Operation step execution time offset list

assemblage_name_dict['2'] = dict()
assemblage_name_dict['2']['name'] = "电源开关组件"
assemblage_name_dict['2']['asm'] = "4,5"
assemblage_name_dict['2']['Reusable devices'] = "涤棉复合切指手套"
assemblage_name_dict['2']['Non-reusable devices'] = "电源开关,条形连接器,电源端子"
assemblage_name_dict['2']['asm-Reusable devices'] = ["电源开关,涤锦符合切指手套", "电源开关,涤锦符合切指手套"]
assemblage_name_dict['2']['asm-Non-reusable devices'] = ["电源端子", "条形连接器"]
assemblage_name_dict['2']['total time'] = "11.26"
assemblage_name_dict['2']['total time variance'] = "2"
assemblage_name_dict['2']['Station number'] = "002"
assemblage_name_dict['2']['order list'] = ["4,5", "5,4"]
assemblage_name_dict['2']['time list'] = ["5.3,5.8", "6.1,5.2"]
assemblage_name_dict['2']['time variance list'] = ["0.5,0.6", "0.6,0.5"]

assemblage_name_dict['3'] = dict()
assemblage_name_dict['3']['name'] = "液晶屏上线"
assemblage_name_dict['3']['asm'] = "6,7,8,9"
assemblage_name_dict['3']['Reusable devices'] = "防静电腕带,涤棉复合全指手套,胶带切割机"
assemblage_name_dict['3']['Non-reusable devices'] = "液晶屏,高速信号线,耐高温胶带*0.1"
assemblage_name_dict['3']['asm-Reusable devices'] = ["防静电腕带,涤棉复合全指手套", "防静电腕带,涤棉复合全指手套", "防静电腕带,涤棉复合全指手套",
                                                     "防静电腕带,涤棉复合全指手套"]
assemblage_name_dict['3']['asm-Non-reusable devices'] = ["", "液晶屏", "", "胶带切割器,高速信号线,耐高温胶带"]
assemblage_name_dict['3']['total time'] = "27.59"
assemblage_name_dict['3']['total time variance'] = "2"
assemblage_name_dict['3']['Station number'] = "001"
assemblage_name_dict['3']['order list'] = ["6,7,8,9"]
assemblage_name_dict['3']['time list'] = ["5.0,6.0,5.1,11.0"]
assemblage_name_dict['3']['time variance list'] = ["1.0,1.1,0.8,2.1"]

assemblage_name_dict['4'] = dict()
assemblage_name_dict['4']['name'] = "主板准备"
assemblage_name_dict['4']['asm'] = "10"
assemblage_name_dict['4']['Reusable devices'] = "防静电腕带,涤棉复合全指手套"
assemblage_name_dict['4']['Non-reusable devices'] = "主板组件,塑料端子板*2"
assemblage_name_dict['4']['asm-Reusable devices'] = ["防静电腕带,涤棉复合全指手套"]
assemblage_name_dict['4']['asm-Non-reusable devices'] = ["主板组件,塑料端子板*2"]
assemblage_name_dict['4']['total time'] = "24.81"
assemblage_name_dict['4']['total time variance'] = "2"
assemblage_name_dict['4']['Station number'] = "002"
assemblage_name_dict['4']['order list'] = ["10"]
assemblage_name_dict['4']['time list'] = ["24.8"]
assemblage_name_dict['4']['time variance list'] = ["2.0"]

assemblage_name_dict['5'] = dict()
assemblage_name_dict['5']['name'] = "安装主板组件"
assemblage_name_dict['5']['asm'] = "11,12"
assemblage_name_dict['5']['Reusable devices'] = "电动螺刀,涤棉复合全指手套,防静电腕带"
assemblage_name_dict['5']['Non-reusable devices'] = "螺钉*6"
assemblage_name_dict['5']['asm-Reusable devices'] = ["电动螺刀,涤棉复合全指手套,防静电腕带", "电动螺刀,涤棉复合全指手套,防静电腕带"]
assemblage_name_dict['5']['asm-Non-reusable devices'] = ["", "螺钉*6"]
assemblage_name_dict['5']['total time'] = "29.31"
assemblage_name_dict['5']['total time variance'] = "2"
assemblage_name_dict['5']['Station number'] = "003"
assemblage_name_dict['5']['order list'] = ["11,12"]
assemblage_name_dict['5']['time list'] = ["3.8,25.0"]
assemblage_name_dict['5']['time variance list'] = ["0.5,2.8"]

assemblage_name_dict['6'] = dict()
assemblage_name_dict['6']['name'] = "安装扬声器组件"
assemblage_name_dict['6']['asm'] = "13"
assemblage_name_dict['6']['Reusable devices'] = "电动螺刀,涤棉复合全指手套,防静电腕带"
assemblage_name_dict['6']['Non-reusable devices'] = "螺钉*4, 内置音箱"
assemblage_name_dict['6']['asm-Reusable devices'] = ["电动螺刀,涤棉复合全指手套,防静电腕带"]
assemblage_name_dict['6']['asm-Non-reusable devices'] = ["螺钉*4, 内置音箱"]
assemblage_name_dict['6']['total time'] = "26.72"
assemblage_name_dict['6']['total time variance'] = "2"
assemblage_name_dict['6']['Station number'] = "004"
assemblage_name_dict['6']['order list'] = ["13"]
assemblage_name_dict['6']['time list'] = ["26.1"]
assemblage_name_dict['6']['time variance list'] = ["2.0"]

assemblage_name_dict['7'] = dict()
assemblage_name_dict['7']['name'] = "安装遥控按键组件"
assemblage_name_dict['7']['asm'] = "14,15,16"
assemblage_name_dict['7']['Reusable devices'] = "扫码枪,涤棉复合全指手套,防静电腕带"
assemblage_name_dict['7']['Non-reusable devices'] = "准备好的按键-遥控组件,条码标签*2"
assemblage_name_dict['7']['asm-Reusable devices'] = ["涤棉复合全指手套,防静电腕带", "涤棉复合全指手套,防静电腕带", "涤棉复合全指手套,防静电腕带,扫码枪"]
assemblage_name_dict['7']['asm-Non-reusable devices'] = ["准备好的按键-遥控组件", "条码标签*2", ""]
assemblage_name_dict['7']['total time'] = "26.21"
assemblage_name_dict['7']['total time variance'] = "2"
assemblage_name_dict['7']['Station number'] = "005"
assemblage_name_dict['7']['order list'] = ["14,15,16", "15,16,14"]
assemblage_name_dict['7']['time list'] = ["16.3,6.0,4.1", "5.8,4.2,15.9"]
assemblage_name_dict['7']['time variance list'] = ["1.5,0.6,0.4", "0.6,0.5,1.4"]

assemblage_name_dict['8'] = dict()
assemblage_name_dict['8']['name'] = "安装电源开关组件"
assemblage_name_dict['8']['asm'] = "17,18"
assemblage_name_dict['8']['Reusable devices'] = "电动螺刀,涤棉复合全指手套,防静电腕带"
assemblage_name_dict['8']['Non-reusable devices'] = "准备好的电源开关组件,电源插座支架,螺钉"
assemblage_name_dict['8']['asm-Reusable devices'] = ["电动螺刀,涤棉复合全指手套,防静电腕带", "涤棉复合全指手套,防静电腕带"]
assemblage_name_dict['8']['asm-Non-reusable devices'] = ["电源插座支架,螺钉", "准备好的电源开关组件"]
assemblage_name_dict['8']['total time'] = "22.42"
assemblage_name_dict['8']['total time variance'] = "2"
assemblage_name_dict['8']['Station number'] = "006"
assemblage_name_dict['8']['order list'] = ["17,18", "18,17"]
assemblage_name_dict['8']['time list'] = ["11.0,12.0", "11.5, 11.5"]
assemblage_name_dict['8']['time variance list'] = ["1.1,1.3", "1.1,1.1"]

assemblage_name_dict['9'] = dict()
assemblage_name_dict['9']['name'] = "插线"
assemblage_name_dict['9']['asm'] = "19"
assemblage_name_dict['9']['Reusable devices'] = "涤棉复合全指手套,防静电腕带"
assemblage_name_dict['9']['Non-reusable devices'] = ""
assemblage_name_dict['9']['asm-Reusable devices'] = ["涤棉复合全指手套,防静电腕带"]
assemblage_name_dict['9']['asm-Non-reusable devices'] = [""]
assemblage_name_dict['9']['total time'] = "23.17"
assemblage_name_dict['9']['total time variance'] = "2"
assemblage_name_dict['9']['Station number'] = "007"
assemblage_name_dict['9']['order list'] = ["19"]
assemblage_name_dict['9']['time list'] = ["23.1"]
assemblage_name_dict['9']['time variance list'] = ["2.0"]

assemblage_name_dict['10'] = dict()
assemblage_name_dict['10']['name'] = "理线"
assemblage_name_dict['10']['asm'] = "20,21"
assemblage_name_dict['10']['Reusable devices'] = "涤棉复合全指手套,防静电腕带"
assemblage_name_dict['10']['Non-reusable devices'] = "耐高温胶带"
assemblage_name_dict['10']['asm-Reusable devices'] = ["涤棉复合全指手套,防静电腕带", "涤棉复合全指手套,防静电腕带"]
assemblage_name_dict['10']['asm-Non-reusable devices'] = ["耐高温胶带", ""]
assemblage_name_dict['10']['total time'] = "30.77"
assemblage_name_dict['10']['total time variance'] = "2"
assemblage_name_dict['10']['Station number'] = "008"
assemblage_name_dict['10']['order list'] = ["20,21"]
assemblage_name_dict['10']['time list'] = ["25.0,5.3"]
assemblage_name_dict['10']['time variance list'] = ["2.1,0.6"]

assemblage_name_dict['11'] = dict()
assemblage_name_dict['11']['name'] = "后壳准备"
assemblage_name_dict['11']['asm'] = "22,23"
assemblage_name_dict['11']['Reusable devices'] = "电动螺刀,涤棉复合切指手套,防静电腕带"
assemblage_name_dict['11']['Non-reusable devices'] = "后壳组件,铭牌,标牌*2"
assemblage_name_dict['11']['asm-Reusable devices'] = ["电动螺刀,涤棉复合切指手套,防静电腕带,后壳组件", "电动螺刀,涤棉复合切指手套,防静电腕带,后壳组件"]
assemblage_name_dict['11']['asm-Non-reusable devices'] = ["标牌*2", "铭牌"]
assemblage_name_dict['11']['total time'] = "26.3"
assemblage_name_dict['11']['total time variance'] = "2"
assemblage_name_dict['11']['Station number'] = "009"
assemblage_name_dict['11']['order list'] = ["22,23", "23,22"]
assemblage_name_dict['11']['time list'] = ["16.0,10.3", "11.0,14.3"]
assemblage_name_dict['11']['time variance list'] = ["1.6,1.1", "1.1,1.5"]

assemblage_name_dict['12'] = dict()
assemblage_name_dict['12']['name'] = "安装后壳(1)"
assemblage_name_dict['12']['asm'] = "24"
assemblage_name_dict['12']['Reusable devices'] = "涤棉复合全指手套,电动螺刀"
assemblage_name_dict['12']['Non-reusable devices'] = "螺钉*6"
assemblage_name_dict['12']['asm-Reusable devices'] = ["涤棉复合全指手套,电动螺刀"]
assemblage_name_dict['12']['asm-Non-reusable devices'] = ["螺钉*6"]
assemblage_name_dict['12']['total time'] = "24.86"
assemblage_name_dict['12']['total time variance'] = "2"
assemblage_name_dict['12']['Station number'] = "010"
assemblage_name_dict['12']['order list'] = ["24"]
assemblage_name_dict['12']['time list'] = ["24.86"]
assemblage_name_dict['12']['time variance list'] = ["2.1"]

assemblage_name_dict['13'] = dict()
assemblage_name_dict['13']['name'] = "安装后壳(2)"
assemblage_name_dict['13']['asm'] = "25"
assemblage_name_dict['13']['Reusable devices'] = "涤棉复合全指手套,电动螺刀"
assemblage_name_dict['13']['Non-reusable devices'] = "螺钉*6"
assemblage_name_dict['13']['asm-Reusable devices'] = ["涤棉复合全指手套,电动螺刀"]
assemblage_name_dict['13']['asm-Non-reusable devices'] = ["螺钉*6"]
assemblage_name_dict['13']['total time'] = "24.86"
assemblage_name_dict['13']['total time variance'] = "2"
assemblage_name_dict['13']['Station number'] = "011"
assemblage_name_dict['13']['order list'] = ["25"]
assemblage_name_dict['13']['time list'] = ["24.86"]
assemblage_name_dict['13']['time variance list'] = ["2.1"]

assemblage_name_dict['14'] = dict()
assemblage_name_dict['14']['name'] = "外观检"
assemblage_name_dict['14']['asm'] = "26,27,28,29,30,31,32"
assemblage_name_dict['14']['Reusable devices'] = "涤棉复合全指手套,擦拭布,塞尺"
assemblage_name_dict['14']['Non-reusable devices'] = "序列号标签小联"
assemblage_name_dict['14']['asm-Reusable devices'] = ["涤棉复合全指手套", "涤棉复合全指手套", "涤棉复合全指手套", "涤棉复合全指手套", "涤棉复合全指手套",
                                                      "涤棉复合全指手套,擦拭布", "涤棉复合全指手套,塞尺"]
assemblage_name_dict['14']['asm-Non-reusable devices'] = ["", "序列号标签小联", "", "", "", "", ""]
assemblage_name_dict['14']['total time'] = "23.41"
assemblage_name_dict['14']['total time variance'] = "2"
assemblage_name_dict['14']['Station number'] = "012"
assemblage_name_dict['14']['order list'] = ["26,27,28,29,30,31,32", "26,27,28,30,29,31,32", "26,27,29,28,30,31,32",
                                            "26,27,29,30,28,31,32", "26,27,30,28,29,31,32", "26,27,30,29,28,31,32"]
assemblage_name_dict['14']['time list'] = ["5.0,2.0,4.0,2.0,3.0,4.0,4.0", "5.0,2.0,4.0,3.0,2.0,4.0,4.0",
                                           "5.0,2.0,2.0,4.0,3.0,4.0,4.0", "5.0,2.0,2.0,3.0,4.0,4.0,4.0",
                                           "5.0,2.0,3.0,4.0,2.0,4.0,4.0", "5.0,2.0,3.0,2.0,4.0,4.0,4.0"]
assemblage_name_dict['14']['time variance list'] = ["0.5,0.2,0.6,0.5,0.3,0.8,0.7", "0.5,0.2,0.6,0.5,0.3,0.6,0.6",
                                                    "0.5,0.2,0.6,0.6,0.5,0.8,0.8", "0.5,0.2,0.4,0.5,0.6,0.6,0.5",
                                                    "0.5,0.4,0.6,0.6,1.0,1.0,0.7", "0.5,0.2,0.5,0.5,0.6,0.6,0.8"]


def toTimeString(timeStamp):
    if type(timeStamp) == int:
        getMS = False
    else:
        getMS = True
    timeTuple = datetime.datetime.utcfromtimestamp(timeStamp + 8 * 3600)
    return timeTuple.strftime(f'%Y-%m-%d %H:%M:%S{r".%f" if getMS else ""}')


# Convert the time string to a 10-digit timestamp, the time string defaults to 2017-10-01 13:37:04 format
def date_to_timestamp(date, format_string="%Y-%m-%d %H:%M:%S.%f"):
    time_array = time.strptime(date, format_string)
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def time_analysis(time):
    # '2020-09-09 15:03:56.00'
    day = time.split(" ")[0]
    day = day.split("-")[-1]
    others = time.split(" ")[1]
    hour = others.split(":")[0]
    if int(day) / 6 == 1.0 or int(day) / 7 == 1.0:
        return "third half"
    if int(hour) > 12:
        return "second half"
    elif int(hour) <= 12:
        return "first half"


def log_check(dataframe):
    print("checking log -----------------------------------------------------")
    all_activity = numpy.array(dataframe["Activity_name"].value_counts().keys())
    for activity in all_activity:
        df = dataframe.loc[dataframe['Activity_name'] == activity]
        print("activity is : ", activity)
        S_t = list()
        E_t = list()
        S_t_string = list()
        E_t_string = list()
        for i, row in df.iterrows():
            S_t.append(date_to_timestamp(row['Start_time']))
            E_t.append(date_to_timestamp(row['End_time']))
            S_t_string.append(row['Start_time'])
            E_t_string.append(row['End_time'])
        for i in range(len(S_t) - 1):
            if E_t[i] > S_t[i + 1]:
                print(i, "wrong")
                print(E_t_string[i], S_t_string[i + 1])
    print("finish checking---------------------------------------------------")

def f_to_c(fine_dataframe):
    trace_id = ""
    Sub_process_name = ""
    rd = []
    nrd = []
    st = ""
    et = ""
    tc = ""
    wn = ""
    sn = ""
    trace_id_list = []
    Sub_process_name_list = []
    rd_list = []
    nrd_list = []
    st_list = []
    et_list = []
    tc_list = []
    wn_list = []
    sn_list = []
    for index, row in fine_dataframe.iterrows():
        # first row
        if trace_id == "":
            trace_id = row['Trace_ID']
            Sub_process_name = row['Sub_process_name']
            rd.append(row['Reusable_devices'])
            nrd.append(row['Non_reusable_devices'])
            st = row['Start_time']
            et = row['End_time']
            wn = row['Worker_name']
            sn = row['Station_number']
        elif row['Sub_process_name'] == Sub_process_name:
            rd.append(row['Reusable_devices'])
            nrd.append(row['Non_reusable_devices'])
            et = row['End_time']
        elif row['Sub_process_name'] != Sub_process_name:
            # finish pre item
            tc = date_to_timestamp(et) - date_to_timestamp(st)
            trace_id_list.append(trace_id)
            Sub_process_name_list.append(Sub_process_name)
            st_list.append(st)
            et_list.append(et)
            tc_list.append(tc)
            wn_list.append(wn)
            sn_list.append(sn)

            rd_value = ""
            for it in list(set(rd)):
                rd_value = rd_value + it + ","
            nrd_value = ""
            for it in list(set(nrd)):
                nrd_value = nrd_value + it + ","
            rd_list.append(rd_value)
            nrd_list.append(nrd_value)

            # as next item start
            trace_id = row['Trace_ID']
            Sub_process_name = row['Sub_process_name']
            rd = []
            nrd = []
            rd.append(row['Reusable_devices'])
            nrd.append(row['Non_reusable_devices'])
            st = row['Start_time']
            et = row['End_time']
            wn = row['Worker_name']
            sn = row['Station_number']

    # last item
    tc = date_to_timestamp(et) - date_to_timestamp(st)
    trace_id_list.append(trace_id)
    Sub_process_name_list.append(Sub_process_name)
    st_list.append(st)
    et_list.append(et)
    tc_list.append(tc)
    wn_list.append(wn)
    sn_list.append(sn)
    rd_value = ""
    for it in list(set(rd)):
        rd_value = rd_value + it + ","
    nrd_value = ""
    for it in list(set(nrd)):
        nrd_value = nrd_value + it + ","
    rd_list.append(rd_value)
    nrd_list.append(nrd_value)

    dataframe = pd.DataFrame({'Trace_ID': trace_id_list, 'Sub_process_name': Sub_process_name_list, 'Sub_process_reusable_devices': rd_list,
                              'Sub_process_non_reusable_devices': nrd_list,
                              'Start_time': st_list, 'End_time': et_list, 'Time_consuming': tc_list, 'Worker_name': wn_list,
                              'Station_number': sn_list})
    return dataframe


# input
start_time_date = "2020-09-15 14:02:01.3"
csv_file = "../data/Activity_level_log_20.csv"
csv_file_coarse = "../data/Process_level_log_20.csv"
trace_num = 300
# The local process interval, simulating the idea of the assembly line, the interval between each station is about
# lpi_s-lpi_e seconds, and a new round of product processing is carried out.
lpi_s = 48
lpi_e = 51
abnormal_ratio = 0.01  # for trace
# 0, 1, 3, 5
abnormal_position_list = [3, 5]
start_time_stamp = date_to_timestamp(start_time_date)
sub_process_orders_temp = assemblage_name_dict['Sub-process orders']
Sub_process_intervals_temp = assemblage_name_dict['Sub-process intervals']
sub_process_orders = list()
Sub_process_intervals = list()

# log information
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

# From the existing work sequence, randomly generate trace_num traces and blank time intervals
for tn in range(trace_num):
    random_index = random.randint(0, len(sub_process_orders_temp) - 1)
    sub_process_orders.append(sub_process_orders_temp[random_index])
    Sub_process_intervals.append(Sub_process_intervals_temp[random_index])

# trace generation
for i in range(len(sub_process_orders)):  # i as current trace num
    sub_processes = sub_process_orders[i].split(",")  # Process list
    Sub_process_interval = Sub_process_intervals[i].split(",")  # Conveyor transfer time between processes
    # simulating the idea of the assembly line, the interval between each station is about
    # lpi_s-lpi_e seconds, and a new round of product processing is carried out.
    start_time_stamp += random.uniform(lpi_s, lpi_e)
    timestamp = start_time_stamp

    abnormal_value = random.random()
    if abnormal_value < abnormal_ratio:
        abnormal_position = random.randint(0, len(abnormal_position_list)-1)
        sub_processes = sub_processes[:abnormal_position_list[abnormal_position]]
        Sub_process_interval = Sub_process_interval[:abnormal_position_list[abnormal_position]]
        print("abnormal at trace:  ", i)
        print(sub_processes)
        print(Sub_process_interval)

    timestamp = float('%.02f' % timestamp)  # two decimal places
    print("trace-%s start time is : %s" % (str(i), toTimeString(timestamp)[:-4]))
    #  Traverse the list of process steps
    for j, sub_process in enumerate(sub_processes):
        sub_process_name = assemblage_name_dict[sub_process]['name']
        station_number = assemblage_name_dict[sub_process]['Station number']
        order_list = assemblage_name_dict[sub_process]['order list']  # Execution sequence list of process steps
        order_index = random.randint(0, len(order_list) - 1)  # Randomly select one execution from the list of
        # process steps execution order
        order = order_list[order_index].split(",")  # example: [22, 23]/ [23, 22]
        avg_ex_times = assemblage_name_dict[sub_process]['time list'][order_index].split(
            ",")  # Average execution time from operation steps, example: [16.0,10.3]
        variance_ex_times = assemblage_name_dict[sub_process]['time variance list'][order_index].split(
            ",")  # Operation step execution time offset
        asm = assemblage_name_dict[sub_process]['asm']  # Get a list of process steps ,example: "22,23"
        asm = asm.split(",")  # ['22', '23']
        asm_Reusable_devices = assemblage_name_dict[sub_process][
            'asm-Reusable devices']
        asm_Non_reusable_devices = assemblage_name_dict[sub_process]['asm-Non-reusable devices']

        # Add conveyor belt time loss for different operations
        timestamp = timestamp + float(Sub_process_interval[j])
        for k, activity in enumerate(order):
            activity_name = assemblage_label[activity]  # step/activity name
            asm_index = 0
            # Find the index position of the current step in the process step list
            for asm_index_ in range(len(asm)):
                if activity == asm[asm_index_]:
                    asm_index = asm_index_
                    break
            # Obtain the reusable and non-reusable resources of the current step according to the index position
            #  Reusable_devices && Non_reusable_devices
            Reusable_devices = asm_Reusable_devices[asm_index]
            Non_reusable_devices = asm_Non_reusable_devices[asm_index]
            # start time
            # Steps between the same process, with random numbers as the step interval
            start_timestamp = timestamp + random.uniform(0.5, 1.0)
            start_time = toTimeString(start_timestamp)
            # end time
            # step evg exc time
            avg_ex_time = avg_ex_times[k]
            # step exc time variance
            variance_ex_time = variance_ex_times[k]
            # Current step execution time = evg + variance
            activity_variance = random.uniform(0 - float(variance_ex_time), float(variance_ex_time)) + float(
                avg_ex_time)
            activity_variance = round(activity_variance, 2)
            status_hour = time_analysis(start_time)
            # Determine worker based on current time
            if status_hour == "first half":
                worker_idx = 0
            elif status_hour == "second half":
                worker_idx = 1
            elif status_hour == "third half":
                worker_idx = 2
            else:
                worker_idx = 0
            worker_name = assemblage_Station_worker_name[assemblage_name_dict[sub_process]['Station number']][
                worker_idx]
            # Current step execution time = Current step execution time/ worker efficiency
            activity_variance = float('%.02f' % (activity_variance / float(worker_work_efficiency[worker_name])))
            end_timestamp = start_timestamp + activity_variance
            end_time = toTimeString(end_timestamp)
            timestamp = end_timestamp
            item = "trace-" + str(
                i) + " " + sub_process_name + " " + activity_name + " " + Reusable_devices + " " + Non_reusable_devices + \
                   " " + start_time[:-4] + " " + end_time[:-4] + " " + format(activity_variance,
                                                                              '.2f') + " " + worker_name
            print(item)
            trace_id.append(str(i))
            sp_name.append(sub_process_name)
            an.append(activity_name)
            Rd.append(Reusable_devices)
            NRd.append(Non_reusable_devices)
            s_time.append(start_time[:-4])
            e_time.append(end_time[:-4])
            atv.append(format(activity_variance, '.2f'))
            wn.append(worker_name)
            Sn.append(station_number)

# convert data into csv with dataframe
dataframe = pd.DataFrame(
    {'Trace_ID': trace_id, 'Sub_process_name': sp_name, 'Activity_name': an, 'Reusable_devices': Rd,
     'Non_reusable_devices': NRd,
     'Start_time': s_time, 'End_time': e_time, 'Time_consuming': atv, 'Worker_name': wn, 'Station_number': Sn})
log_check(dataframe=dataframe)
coarse_dataframe = f_to_c(dataframe)

dataframe.to_csv(csv_file, index=False, sep=',', encoding='utf_8_sig')
coarse_dataframe.to_csv(csv_file_coarse, index=False, sep=',', encoding='utf_8_sig')

