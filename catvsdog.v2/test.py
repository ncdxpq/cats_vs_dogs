from keras.models import load_model
from dataset import *
import matplotlib.pyplot as plt

model_load = load_model("./cats_and_dogs_small_1.h5")
# 读取测试集中的数据
test_dir = r'F:\猫狗大战_01\dogs-vs-cats\test1'
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 归一化
# 所有图像调整为 (150,150) # 因为使用了 binary_crossentropy损失，所以需要用二进制标签 # 批量大小为 20
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
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
model_load.evaluate(data_batch, labels_batch, batch_size=20, verbose=1)
history = model_load.evaluate_generator(test_generator, steps=20)


# def plot_acc_err(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epochs = range(1, len(acc) + 1)
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#
#     plt.figure()  # 在另一个图像绘制
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#     plt.show()
#
#
# plot_acc_err(history)
