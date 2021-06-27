import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing import image  # 数据处理

original_dataset_dir = r'F:\猫狗大战_01\dogs-vs-cats\train'  # 导入数据集
# 创建目录
base_dir = './cats_and_dogs_small'  # 当目录已存在时，无法创建该目录
# os.mkdir(base_dir)  # 创建目录cats_and_dogs_small
train_dir = os.path.join(base_dir, 'train')  # 多个路径组合并且返回，训练集
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')  # 多个路径组合并且返回，测试集
# s.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')  # 验证集
# os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
# 将图像复制到训练、验证和测试目录中
# fnames = ['cat.{}.jpg'.format(i) for i in range(3000)]  # 3000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(train_cats_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到训练集里的猫图像
# fnames = ['cat.{}.jpg'.format(i) for i in range(3000, 4000)]  # 1000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(validation_cats_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到验证集里的猫图像
# fnames = ['cat.{}.jpg'.format(i) for i in range(4000, 5000)]  # 1000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(test_cats_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到测试集里的猫图像
# fnames = ['dog.{}.jpg'.format(i) for i in range(3000)]  # 3000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(train_dogs_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到训练集里的狗图像
# fnames = ['dog.{}.jpg'.format(i) for i in range(3000, 4000)]  # 1000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(validation_dogs_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到验证集里的狗图像
# fnames = ['dog.{}.jpg'.format(i) for i in range(4000, 5000)]  # 1000张
# for i in fnames:
#     scr = os.path.join(original_dataset_dir, i)
#     dst = os.path.join(test_dogs_dir, i)
#     shutil.copyfile(scr, dst)  # 复制得到测试集里的狗图像
# print('数据集准备成功！')
# 输出各数据集的图片数量
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
# 归一化
train_datagen = ImageDataGenerator(rescale=1. / 255)  # 将所有图像乘以 1./255 进行缩放，即进行归一化
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = r'F:\猫狗大战_01\cats_and_dogs_small\train'  # 训练集路径
validation_dir = r'F:\猫狗大战_01\cats_and_dogs_small\validation'  # 验证集路径

# 所有图像调整为 (150,150) # 因为使用了 binary_crossentropy损失，所以需要用二进制标签 # 批量大小为 20
# 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
# class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical.
# 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签.
# "sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据,
# 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=30, class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=30, class_mode='binary')
# 对数据进行可视化查看，看图片和标签是否匹配，并且随机检查一部分
for data_batch, label_batch in train_generator:
    x = data_batch[0]
    y = label_batch[0]
    x_1 = data_batch[1]
    print("data_batch shape", data_batch.shape)
    print("lable_batch shape", label_batch.shape)
    break
plt.imshow(image.array_to_img(x))
plt.title(str('cat' if y == 0 else 'dog'))
plt.show()
