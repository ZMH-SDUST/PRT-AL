# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/13 8:51
@Auther ： Zzou
@File ：xes_to_csv.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import pandas as pd
import pm4py
from numpy import sort

file_path = "Event Log.xes"
log = pm4py.read_xes(file_path)
pd_data = pm4py.convert_to_dataframe(log)

print("number of columns is :", str(len(pd_data.columns)))
print(sort(pd_data.columns))
for col in pd_data.columns:
    print(col, len(pd_data[col].unique()))

# pd_data.to_csv("2021.csv", encoding="utf-8")
