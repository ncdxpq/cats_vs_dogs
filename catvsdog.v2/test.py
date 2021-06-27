from keras.models import load_model
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  # 中文字体需要导入

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置可以显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
model_load = load_model("./cats_and_dogs_small_1.h5")
# 读取测试集中的数据
test_dir = r'F:\猫狗大战_01\cats_and_dogs_small\test'
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 归一化
# 所有图像调整为 (150,150) # 因为使用了 binary_crossentropy损失，所以需要用二进制标签 # 批量大小为 50
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=50,  # 2000张，2000/50=40
                                                  class_mode='binary')
# 可视化部分图像，看图像与标签是否相符
for data_batch, labels_batch in test_generator:
    x = data_batch[0]
    y = labels_batch[1]
    print("data_batch shape", data_batch.shape)
    print("labels_batch shape", labels_batch.shape)
    break

plt.imshow(image.array_to_img(x))
plt.title(str('cat' if y == 0 else 'dog'))
plt.show()
# 对测试集进行预测
model_load = load_model("./cats_and_dogs_small_1.h5")
# model_load.evaluate(data_batch, labels_batch, batch_size=20, verbose=1)

test_loss, test_acc = model_load.evaluate(test_generator, steps=50)
print('test acc:', test_acc)
print('test loss:', test_loss)

