import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras import models
import matplotlib.pyplot as plt

# streamlit上での警告を表示しないようにする
st.set_option('deprecation.showPyplotGlobalUse', False)
# 学習済みモデル「cats_and_dogs_small_1.h5」を用いる
model = models.load_model('cats_and_dogs_small_1.h5')

# チェックボックスの種類
cats_dogs_var = st.selectbox("ネコ・イヌの種類",("cat1", "cat2", "cat3", "dog1", "dog2", "dog3"))

# 可視化する画像を選択
if cats_dogs_var == "cat1":
    img_path = 'dog_cat_img/cat.1.jpg'

elif cats_dogs_var == "cat2":
    img_path = 'dog_cat_img/cat.2.jpg'

elif cats_dogs_var == "cat3":
    img_path = 'dog_cat_img/cat.3.jpg'

elif cats_dogs_var == "dog1":
    img_path = 'dog_cat_img/dog.1.jpg'
    
elif cats_dogs_var == "dog2":
    img_path = 'dog_cat_img/dog.2.jpg'
    
elif cats_dogs_var == "dog3":
    img_path = 'dog_cat_img/dog.3.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.
plt.imshow(img_tensor[0])

# オリジナル画像を表示
st.pyplot()

# 出力側の8つの層から出力を抽出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 特定の入力をもとに、これらの出力を返すモデルを作成
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# 5つのNumpy配列(層の活性化ごとに1つ)のリストを返す
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

# 最初の層の活性化の3番目のチャネル
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
st.pyplot()

# プロットの一部として使用する層の名前
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# 特徴マップを表示
for layer_name, layer_activation in zip(layer_names, activations):
    # 特徴マップに含まれている特徴量の数
    n_features = layer_activation.shape[-1]

    # 特徴マップの形状 (1, size, size, n_features)
    size = layer_activation.shape[1]

    # この行列で活性化チャネルをタイル表示
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 各フィルタを1つの大きな水平グリッドでタイル表示
    for col in range(n_cols):
         for row in range(images_per_row):
            channel_image = layer_activation[0,
                                            :, :,
                                            col * images_per_row + row]
            # 特徴量の見た目をよくするための後処理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image

    # グリッド表示
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    st.pyplot()
