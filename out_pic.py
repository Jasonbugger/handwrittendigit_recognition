# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:33:32 2017

@author: Jason Bug

以训练集图片为例，对图片处理结果储存，观测效果
"""

from handwritten import *
import scipy.misc



base_path = "image/train/"
for i in range(1000):
    name = str(i)+".jpg"
    number = int(number_list[counter].split(' ')[1])

    pic1 = Get_Bgraph(base_path + name)
    pic2 = Del_Dirt(pic1)
    higher = Get_Highest_Point(pic2)
    lower = Get_Lowest_Point(pic2)
    pic = rotate(pic2, higher, lower)
    pic = Xihua(pic,array,4)                #array在handwritten文件中
    
    
    pic[pic==1] = 255
    scipy.misc.imsave("image/change/"+name,pic)
    
#结果输出经过二值化，去污点，倾斜矫正，细化的图片