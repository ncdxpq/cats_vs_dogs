import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

dataset_path = r"F:/dogvscat/"
print(os.listdir(dataset_path))

# 图片大小格式设定
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # channel=3彩色（三原色红绿蓝），channel=1黑白

# 整理训练标签
filenames = os.listdir(dataset_path + "train")  # os.listdir()用于返回一个由文件名和目录名组成的列表
categories = []
for filename in filenames:
    category = filename.split('.')[0]  # 文件名按照.分割，并取序列为0的项
    if category == 'dog':
        categories.append(1)  # 在列表末尾添加1代表狗
    else:
        categories.append(0)  # 在列表末尾添加0代表猫

df = pd.DataFrame({  # pandas 二维表
    'filename': filenames,
    'category': categories})
# print(df)查看一下表
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})  # 二维表替换标签
# print(df)再查看也行
# 交叉验证中常用的函数，功能是从样本中随机的按比例选取train_data和test_data
# 训练集因为没传入参数为验证集的补集,验证集下面传了25%，故训练集75%
train_df, validate_df = train_test_split(df, test_size=0.25, random_state=42)
# reset_index用来重置索引，因为有时候对dataframe做处理后索引可能是乱的。
train_df = train_df.reset_index(drop=True)
# drop=True就是把原来的索引index列去掉，再添加重置的index。drop=False就是保留原来的索引，再添加重置的index。
validate_df = validate_df.reset_index(drop=True)

# 显示训练集数据
# print('显示训练集数据')
# train_df['category'].value_counts().plot.bar()
# validate_df['category'].value_counts().plot.bar()

total_train = train_df.shape[0]  # 训练集图片的数量 [0]获取是数据的长度 6003
total_validate = validate_df.shape[0]  # 验证集图片的数量 2002
batch_size = 20

train_datagen = ImageDataGenerator(  # 数据增强生成器，对数据集中的图片进行旋转尺寸大小等变换
    rotation_range=15,  # 旋转角度
    rescale=1. / 255,  # 将所有图像乘以 1./255 进行缩放，即进行归一化
    shear_range=0.1,  # 随机错切变化的角度
    zoom_range=0.2,  # 随机缩放的范围
    horizontal_flip=True,  # 随机将一半图像水平翻转=True
    width_shift_range=0.1,  # 左右平移
    height_shift_range=0.1  # 上下平移
)
# 配置训练集生成器，用于feed模型
train_generator = train_datagen.flow_from_dataframe(  # 输入dataframe和目录的路径
    train_df,
    dataset_path + "train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# 带来验证集数据生成器
validation_datagen = ImageDataGenerator(rescale=1. / 255)  # 归一化
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    dataset_path + "train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
# 测试集
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    dataset_path + "test/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

'''
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

sample = random.choice(filenames)
image = load_img(dataset_path + "train/" + sample)
plt.imshow(image)
'''
