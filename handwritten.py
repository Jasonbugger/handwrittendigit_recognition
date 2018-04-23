# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:26:33 2017

@author: Jason Bug
首先将图像用数组进行表示
所用到的库有Numpy，PIL（图像处理）
读取的已经是灰度图
二值化
利用数组运算，对手写数字图片进行处理
图像矫正：通过找图像的最高点和最低点（有多个取中点）获得中心线，将中心线旋转至垂直
手段：利用中心线与垂直线的斜率处理
最后得到的黑白图片，提取黑色点进行特征分析（特征向量等等）

贝叶斯原理：训练过程：数字为n的条件下，表现出Ai特征的概率为xxx，表现出Bi特征的概率为XXX……
            检验过程：在表现出Ai，Bi……特征的条件下，该数字为n的概率为XXX，选择最可能的作为结果

Image.convert('L')可以将图像直接转化为二维灰度图像，省去褪色的操作
然后再进行二值化，得到黑白图片。
之后对黑白图片矩阵进行操作，包括归一化，细化笔画，去毛刺等等
之后对于特征值的选取，需要借鉴已有研究。（多看论文）
*********************************************************
目前达到的最好效果正确率只有55%
必须使用笔画细化算法
"""

from PIL import Image #表示图像（Image，open）
import numpy as np

 #用于笔画细化使用的矩阵
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]
# 二值化部分
def Get_Bgraph(path):
    pic = np.array(Image.open(path))

   # pic[pic >= 200] = 255 # 第一版参数为200时记为1.但后来遇到了笔画非常细的数字，完全不能识别，
   #改进算法，对于高于50的点，直接认定是有效的点，对于其他在50到100的点，要根据周围情况判断
    pic_1 = np.zeros_like(pic)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j]>100:
                pic[i][j] = 1
            elif pic[i][j]>0:
                pic[i][j] = 2
            else:
                pic[i][j] = 0
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if i==0 or i==pic.shape[0]-1 or j==0 or j==pic.shape[1]-1:
                pic_1[i][j] = 0
                continue
            if pic[i][j]==1:
                pic_1[i][j] = 1
            elif pic[i][j]==2:
                counter_1 = 0
                counter_2 = 0
                for a in range(9):                      #对周围点特征的统计
                    if pic[i-1+a//3][j-1+a%3]==0:
                        counter_1 += 1
                    elif pic[i-1+a//3][j-1+a%3]==2:
                        counter_2 += 1                      #不显著点：灰度值在50以下的点。实笔画：灰度值在50以上的点
                if counter_2>6 and counter_1 >= 4:      #在周围不显著点达到一定数量并且实笔画较少的时候，将点化为实笔画
                    pic_1[i][j] = 1
                else:
                    pic_1[i][j] = 0
    return pic_1


""" 创建新的array pic_1，用来储存去除干扰点的矩阵"""
"""去干扰点，补小空"""
#图像细化部分（算法及代码来自网络），赋予某点周围点不同的权值，根据权值在对应的数字中找到该点是否应该被去除
#该点被去除后，下一个点不做改变
def VThin(pic,array):
    h = pic.shape[0]
    w = pic.shape[1]
    NEXT = 1
    for i in range(w):
        for j in range(h):
            pic[i][j] = 1 if pic[i][j]==0 else 0            #反转：网络算法基于笔画为黑色，本例正好相反
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = pic[i][j-1]+pic[i][j]+pic[i][j+1] if 0<j<w-1 else 1
                if pic[i][j] == 0  and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and pic[i-1+k,j-1+l]==1:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    pic[i][j] = array[sum]
                    if array[sum] == 1:
                        NEXT = 0
    for i in range(w):                                      #处理结束，返回笔画为白色的图像
        for j in range(h):
            pic[i][j] = 1 if pic[i][j]==0 else 0
    return pic

    
def HThin(pic,array):
    h = pic.shape[0]
    w = pic.shape[1]
    NEXT = 1
    for i in range(w):
        for j in range(h):
            if pic[i][j]==0:
                pic[i][j] = 1  
            else:
                pic[i][j] = 0
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = pic[i-1][j]+pic[i][j]+pic[i+1][j] if 0<i<h-1 else 1   
                if pic[i][j] == 0 and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and pic[i-1+k][j-1+l]==1:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    pic[i][j] = array[sum]
                    if array[sum] == 1:
                        NEXT = 0
    for i in range(w):
        for j in range(h):
            if pic[i][j]==0:
                pic[i][j] = 1  
            else:
                pic[i][j] = 0
    return pic


def Xihua(pic,array,num=1):
    #做n次迭代处理，实测不用太多次
    for i in range(num):
        pic = VThin(pic,array)
        pic = HThin(pic,array)
    return pic


#去污点部分，统计某处有笔画的点的附近的点的分布情况，但是这方法，不能去除较大的污点
def Del_Dirt(pic):
    pic_1 = np.zeros_like(pic)
    for i in range(1,pic.shape[0]-1):
        for j in range(1,pic.shape[1]-1):           #注意：边界不能这样判断
            if pic[i][j]==1:
                if pic[i-1][j-1]+pic[i][j-1]+pic[i+1][j-1]+pic[i-1][j]+\
                    pic[i][j]+pic[i+1][j]+pic[i-1][j+1]+pic[i][j+1]+\
                    pic[i+1][j+1] <= 3:
                    pic_1[i][j] = 0
                else:
                    pic_1[i][j] = 1
            else:
                if pic[i-1][j-1]+pic[i][j-1]+pic[i+1][j-1]+pic[i-1][j]+\
                    pic[i][j]+pic[i+1][j]+pic[i-1][j+1]+pic[i][j+1]+\
                    pic[i+1][j+1] > 4:
                    pic_1[i][j] = 1
                else:
                    pic_1[i][j] = 0
    return pic_1


"""该方法去除干扰点不够完善，可能保留部分较大的污点
            and pic[i][j+1]==0 and pic[i][j-1]==0 and pic[i+1][j+1]==0\
            and pic[i+1][j-1]==0 and pic[i-1][j+1]==0 and pic[i-1][j-1]==0:
            if pic[i+1][j]==0 and pic[i-1][j]==0\
                pic[i][j] = 0
"""



# 图像倾斜矫正
# 获得最高点坐标（横向平均）
def Get_Highest_Point(pic):
    max_height_j = 0
    max_height_i = 0
    temp_right = 0
    temp_left = 0
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] == 1:
                temp_left = j
                break
        for j in range(pic.shape[1]):
            if pic[i][pic.shape[1]-1-j] == 1:
                temp_right = pic.shape[1]-1-j
                break
        if temp_left != 0:
            max_height_j = (temp_left+temp_right)//2
            max_height_i = i
            break
    return [max_height_i, max_height_j]


# 获得最低点（横向平均）
def Get_Lowest_Point(pic):
    min_height_j = 0
    min_height_i = 0
    temp_right = 0
    temp_left = 0
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[pic.shape[0]-i-1][j] == 1:
                temp_left = j
                break
        for j in range(pic.shape[1]):
            if pic[pic.shape[0]-1-i][pic.shape[1]-1-j] == 1:
                temp_right = pic.shape[1]-1-j
                break
        if temp_left != 0:
            min_height_j = (temp_left+temp_right)//2
            min_height_i = pic.shape[0]-i-1
            break
    return [min_height_i, min_height_j]             #以i，j形式返回


def rotate(pic, max_array, min_array):
    #参数:图像矩阵，最高点坐标，最低点坐标
    mid_location = [(max_array[0]+min_array[0])//2,(max_array[1]+min_array[1])//2]
    if(max_array[0]!=min_array[0]):
        k = (max_array[1]-min_array[1])/(max_array[0]-min_array[0]) # 计算获得倾斜度
    else:
        return pic #如果最高点和最低点的横坐标相同，则不用旋转，直接返回
    pic_result = np.zeros_like(pic)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] == 1:
                if(int(j-(i-mid_location[0])*k)<28 and int(j-(i-mid_location[0])*k)>0):
                    pic_result[i][int(j-(i-mid_location[0])*k)] = 1
    return pic_result




#特征提取部分
#特征尝试3……按小块分割
def Get_block_rate(pic,step=2):
    rate_list = []
    for i in range(0,28,step):
        for j in range(0,28,step):                                                                                                                                                  
            feature = 0
            for a in range(0,step):
                for b in range(0,step):
                    feature += pic[i+a][j+b]
            rate_list.append(feature)
    return rate_list
            
            
# 特征尝试2，将矩阵按行记非空点的比例
def Get_line_rate(pic):
    rate_list = []
    for i in range(28):
        rate_list.append(0)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if(pic[i][j]==1):
                rate_list[i] += 1
        rate_list[i] /= 16
    return rate_list


def Has_cycle(pic,i,j):
    if i>pic.shape[0]-1 or j>pic.shape[1]-1 or i<0 or j<0:
        return 
    else:
        if pic[i][j]== 0:
            pic[i][j] = 2
            Has_cycle(pic,i+1,j)
            Has_cycle(pic,i,j+1)
            Has_cycle(pic,i-1,j)
            Has_cycle(pic,i,j-1)
        
def cycle_feature(pic):
    global pic_1
    pic_1 = pic

    Has_cycle(pic_1,0,0)
    for i in range(pic_1.shape[0]):
        for j in range(pic_1.shape[1]):
            if pic_1[i][j]==0:
                return 1
    return 0
    
# 特征尝试1：将矩阵分为14*14块，对每一个14*14矩阵计算其中的非空白点的比例，



#将比例转化为零一的离散变量
def Get_Parm(rate_list):
    # 特征列表，是一列0-1之间的数，记录了每一块点中的概率
    for i in range(len(rate_list)):
        if rate_list[i] > 0.1:
            rate_list[i] = 1
        else:
            rate_list[i] = 0
    return rate_list



#统计函数部分（利用增量算法计算概率）
#获得数字/某特征出现的概率
def Get_Possiblity(pic_number, total_number, rate_list):
    #参数：图像对应数字/特征列表，总数，数字/特征概率列表
    # 该函数返回一个数组，记录在test集中，所有数字/特征出现的概率
    if type(pic_number) == int:   # 对数字的处理
        for i in range(len(rate_list)):
            if(pic_number == i):
                rate_list[i] = (rate_list[i]*(total_number-1)+1)/total_number
            else:
                rate_list[i] = rate_list[i]*(total_number-1)/total_number
        return rate_list
    elif type(pic_number) == list:  #对特征向量的处理
        for i in range(len(rate_list)):
            if pic_number[i] == 1:
                rate_list[i] = (rate_list[i]*(total_number-1)+1)/total_number
            else:
                rate_list[i] = rate_list[i]*(total_number-1)/total_number
        return rate_list


# 获得P(A|B)的矩阵（10*50）
def Get_Martix(rate_Martix, number, total_number, feature):
    # 参数：概率矩阵，表示在数字Ai条件下，特征Bi的概率，输入的数字，数字总数，特征值
    for i in range(len(rate_Martix[0])):
        if feature[i] == 1:
            rate_Martix[number][i] = ((rate_Martix[number][i]*(total_number-1))+1)/total_number
        else:
            rate_Martix[number][i] = (rate_Martix[number][i]*(total_number-1))/total_number
    return rate_Martix
