import os
import json
import pandas as pd

csv_path = '/home/yzbj10/Work/lxj/data/mini_imagenet/mini-imagenet/new_val.csv'
csv_data = pd.read_csv(csv_path)
total_num = csv_data.shape[0]
img_paths = [i for i in csv_data["filename"].values]
img_label = [i for i in csv_data["label"].values]

# import pdb
# pdb.set_trace()

aaa = []
bbb = []    

dic={}
for i in img_label:
    dic[i]=0

for j in range(len(img_paths)):
    if dic[img_label[j]]<20: #100
        dic[img_label[j]]+=1
        aaa.append(img_paths[j])
        bbb.append(img_label[j])


#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'filename':aaa,'label':bbb})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("/home/yzbj10/Work/lxj/data/mini_imagenet/mini-imagenet/new_val_hk.csv",index=False,sep=',')