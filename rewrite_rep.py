import os
import json
import pandas as pd

csv_path = '/home/yzbj10/Work/lxj/data/mini_imagenet/mini-imagenet/new_train_hk_ori.csv'
csv_data = pd.read_csv(csv_path)
total_num = csv_data.shape[0]
img_paths = [i for i in csv_data["filename"].values]
img_label = [i for i in csv_data["label"].values]

# import pdb
# pdb.set_trace()

aaa = []
bbb = []    

count=0

for j in range(len(img_paths)):
    aaa.append(img_paths[j])
    bbb.append(count)
    count+=1
    if count>=500:
        count=0


#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'filename':aaa,'label':bbb})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("/home/yzbj10/Work/lxj/data/mini_imagenet/mini-imagenet/new_train_500_hk.csv",index=False,sep=',')