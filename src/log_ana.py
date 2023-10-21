import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "../data/Process_level_log_all.csv"
# input_file = "Production_Data.csv"
tips = pd.read_csv(input_file)
tips.head()

tips = tips[(tips['Sub_process_name'] == '安装遥控按键组件')]

worker_id = np.array(tips["Worker_name"].values)
timestamp = np.array(tips['Time_consuming'].values)
Sub_process_name = np.array(tips["Sub_process_name"].values)

work_id_set = np.unique(worker_id)
Sub_process_name_set = np.unique(Sub_process_name)

# convert value to figure
# print(work_id_set)
# for i in range(len(worker_id)):
#     worker_id[i] = np.where(work_id_set == worker_id[i])[0]
# for i in range(len(Sub_process_name)):
#     Sub_process_name[i] = np.where(Sub_process_name_set == Sub_process_name[i])[0]


x = worker_id
y = timestamp

for work in work_id_set:
    print(work)
    data_list = []
    for j in range(len(worker_id)):
        if worker_id[j] == work:
            data_list.append(timestamp[j])
    print(data_list)
    print(max(data_list))
    print(min(data_list))

c = worker_id
for i in np.unique(x):
    plt.axvline(i, color='lightgray')
plt.scatter(x, y, c=c, s=20)
plt.xlabel('worker_id')
plt.ylabel('timestamp')
plt.show()

