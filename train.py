# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:54:14 2017

@author: JasonBug
"""
import os
from pandas import DataFrame
from pandas import Series
from handwritten import *
import time
"""
利用test数字集，得到P(Ai)，P(Bk|Ai)（在数字为某数字的情况下，特征B表现为Bk的概率）
P(Bk)特征B的整体发生概率
""" 

start_time = time.time()    #记录开始时间
path = "image/train/"       #图片根目录
file_list = os.listdir(path)#获得图片列表

rate_feature_list = []      #特征概率(P(Bk))
rate_Martix = [[], [], [], [], [], [], [], [], [], []]  #条件概率矩阵
rate_list = []              #数字概率矩阵

number_path = "image/train.txt"  #图片标签文件路径
txt_file = open(number_path, 'r')
number_list = txt_file.readlines()  #图片标签列表

#生成储存数据需要的矩阵
for i in range(50):
    rate_feature_list.append(0)
for i in range(10):
    rate_list.append(0)
    for j in range(50):
        rate_Martix[i].append(0)

counter = 0
for k in range(30000):                              #遍历图片文件
    i = str(k)+".jpg"
    number = int(number_list[counter].split(' ')[1])#获得txt文件中的答案(标签)
    
    #图片处理
    pic1 = Get_Bgraph(path+i)       #二值化
    pic2 = Del_Dirt(pic1)           #去污点
    higher = Get_Highest_Point(pic2)#获得最高点坐标
    lower = Get_Lowest_Point(pic2)  #获得最低点坐标
    pic = rotate(pic2, higher, lower)#旋转操作
    pic = Xihua(pic,array)          #细化操作

    #提取特征向量
    feature_list = Get_block_rate(pic,4)    #小块特征（49个）
    feature_list.append(cycle_feature(pic)) #添加环特征
    value_list  = Get_Parm(feature_list)    #特征转化为0-1变量
    Get_Possiblity(number, counter+1, rate_list)    #增量算法更新数字概率
    Get_Possiblity(value_list, counter+1, rate_feature_list)#增量算法更新特征概率
    Get_Martix(rate_Martix, number, counter+1, value_list)#增量算法更新条件概率矩阵
    print(k," has finished") 
    counter += 1

#将结果储存在csv文件中
DataFrame_rate_Martix = DataFrame(rate_Martix)
DataFrame_rate_Martix.to_csv("data/Rate_Matrix.csv")
Series_feature_list = Series(rate_feature_list)
Series_feature_list.to_csv("data/Rate_Feature.csv")
Series_number_rate = Series(rate_list)
Series_number_rate.to_csv("data/Number_Rate.csv")

end_time = time.time()
print("time:",end_time-start_time) #输出所需时间