# refer:https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

from keras.models import Sequential  # 顺序模型
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model, save_model
from dataset import *
import os
from matplotlib import pyplot as plt

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
# 创建猫狗分类模型
model = Sequential()  # Sequential模型，Sequential模型的核心操作是添加layers（图层）
# 向模型添加一个带有32个大小为3 * 3的过滤器的卷积层，卷积层获取128 * 128 * 3的输入图像
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# 多层CNN里，BN放在卷积层之后，激活和池化之前，极大提升了训练速度，收敛过程大大加快
model.add(BatchNormalization())
# 最大池化层，有的不填写是因为有默认值。池化核的尺寸，默认是2×2
model.add(MaxPooling2D(pool_size=(2, 2)))
# dropout在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。CNN中防止过拟合提高效果
model.add(Dropout(0.25))
# 再卷积，池化
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())  # Flatten再Dense
# 全连接层
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 二维：只有猫和狗两类

# 编译模型，配置优化器、选择损失函数
# loss损失函数：交叉熵损失函数（categorical_crossentropy）一般配合softmax做标签分类，线性回归等才用平方差损失函数。
# optimizer优化器，使用optimizer优化器
# metrics评估函数，accuracy准确率
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.summary()输出模型各层的参数状况
model.summary()

# 配置模型训练时的函数
# earlystop：提前停止训练的callbacks。达到当训练集上的loss不在减小（即减小的程度小于某个阈值）的时候停止继续训练。
earlystop = EarlyStopping(patience=10)  # patience：能够容忍多少个epoch内都没有improvement。
# 调整学习率
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',  # 被监测的量
                                            patience=2,  # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                            verbose=3,
                                            factor=0.5,  # 每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                            min_lr=0.00001)  # 学习率的下限
# 回调函数
callbacks = [earlystop, learning_rate_reduction]

# 如果已经有预训练模型，就加载进来再训练
if os.path.exists("dog_vs_cat_model.h5"):
    print("reload model...")
    model = load_model("dog_vs_cat_model.h5")

epochs = 18
# 利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
history = model.fit(
    train_generator,  # 训练集生成器函数
    epochs=epochs,  # 把所有数据训练的次数轮数
    validation_data=validation_generator,  # 验证集生成器函数
    validation_steps=total_validate // batch_size,  # 当validation_data为生成器时，本参数指定验证集的生成器返回次数2002/15
    steps_per_epoch=total_train // batch_size,  # 整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch 6003/15=
    callbacks=callbacks
)
# 训练完保存模型
save_model(model, "dog_vs_cat_model.h5")
print('保存模型成功')
# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.set_xticks(np.arange(1, epochs, 1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
