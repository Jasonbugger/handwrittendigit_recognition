from handwritten import *
import time



#将训练得到的数据导入
def Import_Trained_Data(path):
    file_feature_rate = []
    file_number_rate = []
    file_matrix = []
    file = open(path + "Rate_Matrix.csv", 'r')
    file_matrix_list = file.readlines()
    file_matrix_list = file_matrix_list[1:]
    counter = 0
    for i in file_matrix_list:
        file_matrix.append([])
        a = i.split(',')
        a = a[1:]
        for j in a:
            file_matrix[counter].append(float(j))
        counter += 1
    file.close()

    file = open(path + "Number_Rate.csv", 'r')
    file_number_rate_list = file.readlines()
    for i in file_number_rate_list:
        a = i.split(',')
        file_number_rate.append(float(a[1]))
    file.close()

    file = open(path + "Rate_Feature.csv", 'r')
    file_feature_rate_list = file.readlines()
    for i in file_feature_rate_list:
        a = i.split(',')
        file_feature_rate.append(float(a[1]))
    file.close()
    return file_matrix, file_number_rate, file_feature_rate




#导入训练结果
file_matrix, file_number_rate, file_feature_rate = Import_Trained_Data("data/")

#设置初始变量
base_path = "image/val/"
number_path = "F://val.txt" #分别是测试集图片和测试集结果，序号从30000到42000
txt_file = open(number_path, 'r')
number_list = txt_file.readlines() #读入测试结果
right_list = [] #储存正确率的列表
total_number_list = [] #各数字总的数量
counter = 0 #测试的总数字数量
right = 0 # 正确数
wrong = 0 #错误数
for i in range(10):
    right_list.append(0)
    total_number_list.append(0)
start_time = time.time()  #计算时间

#遍历图片，对图像进行处理，得到特征向量
for i in range(30000,32000):
    if(str(i)+".jpg"!=number_list[counter].split(' ')[0]):
        continue
    try:
        number = int(number_list[counter].split(' ')[1])
    except:
        print(number_list[counter])
    pic1 = Get_Bgraph(base_path + str(i)+".jpg")
    pic2 = Del_Dirt(pic1)
    higher = Get_Highest_Point(pic2)
    lower = Get_Lowest_Point(pic2)
    pic = rotate(pic2, higher, lower)
    pic = Xihua(pic,array)
    feature_list = Get_block_rate(pic,4)
    feature_list.append(cycle_feature(pic))
    value_list = Get_Parm(feature_list)
    
    #利用朴素贝叶斯，结合最小犯错原则，对图片进行判定
    max_rate = 0.0
    max_rate_number = 0
    for j in range(len(file_matrix)):         # 第一维j是猜测可能的数字
        rate_A_B = 1 #rate_A_B：在特征B下，数字为A的概率为 rate_A_B，经过下面的迭代
                     #，将得到在所有特征下，数字为A的概率
        for k in range(len(file_matrix[0])):     # 第二维k是待测数字的特征向量

            if value_list[k]==1:
                rate_A_B *= file_matrix[j][k]/file_feature_rate[k]
            else:
                rate_A_B *= (1-file_matrix[j][k])/(1-file_feature_rate[k])
        rate_A_B *=file_number_rate[j]
        if rate_A_B > max_rate:  #若当前概率更大，则选择结果为当前数字
            max_rate = rate_A_B
            max_rate_number = j

# 结果判定
    if number == max_rate_number:
        print("right")
        # 这是判断正确的情况
        right_list[number] = (right_list[number]*total_number_list[number]+1)/(total_number_list[number]+1)
        total_number_list[number] += 1
        right+=1
    else:
        wrong+=1
        right_list[number] = (right_list[number] * total_number_list[number]) / (total_number_list[number] + 1)
        total_number_list[number] += 1
        print("wrong")
    counter += 1

#结果输出
print("right:wrong:",right,wrong) #输出错误正确比例
print("right rate:",right/(right+wrong))
end_time = time.time()
print("time:",end_time-start_time) #输出所需时间
print(right_list) #输出每个数字的识别率