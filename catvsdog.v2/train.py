from dataset import *
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing import image  # 数据处理
from keras import optimizers


def model_demo():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 二分类，最后一层使用 sigmoid
    return model


model = model_demo()
model.summary()
# 编译模型

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),  # 直接使用 RmSprop 是稳妥的。
              loss=keras.losses.binary_crossentropy, metrics=['acc'])  # 二分类所以使用二元交叉熵作为损失函数
# 利用批量生成器拟合模型
# 得出的结果是训练集和验证集上的损失和精度
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=100,
                              epochs=20,  # 训练 20 轮
                              validation_data=validation_generator,
                              validation_steps=50
                              )
# 保存模型，是开发的一种习惯
model.save('cats_and_dogs_small_1.h5')


# 绘图
def plot_acc_err(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()  # 在另一个图像绘制
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


plot_acc_err(history)
