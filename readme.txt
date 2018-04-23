文件
handwritten.py：图片处理和特征值获取的函数
train.py：获得统计数据
test.py：检测
out_pic：输出效果观测
文件夹：
data：储存train.pyu训练后输出的相关概率向量和概率矩阵
image：
	val:检测图片集
	change2：处理效果观测，是由train中的文件经处理获得的
	test：训练集，利用他们获得相关概率


环境配置（python 3.X（本机使用anaconda,python 3.6））
需要安装的库有：numpy，pandas
（开始菜单搜索cmd->管理员权限运行->输入pip install numpy
					结束后输入pip install pandas））
先运行train（已经将结果放入文件夹，运行时间太久，可以不运行）
再运行test，最终可以获得各数字的正确率
	