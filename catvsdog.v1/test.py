import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, save_model
from dataset import *

# 配置测试集数据生成器
test_filenames = os.listdir(dataset_path + "test")
test_df = pd.DataFrame({
    'filename': test_filenames})
print(test_df)
nb_samples = test_df.shape[0]  # 测试集图片数量
test_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_gen.flow_from_dataframe(  # 测试集数据生成器
    test_df,
    dataset_path + "test/",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=True
)

# 加载预训练模型
model = load_model("dog_vs_cat_model.h5")

# 使用预训练模型对测试集进行预测
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))
test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)  # 取前18张图片验证
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(dataset_path + "test/" + filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()
