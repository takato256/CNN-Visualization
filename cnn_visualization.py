import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()


def original_img():
    # 画像を選択
    cats_dogs_var = st.sidebar.selectbox("画像を選択してください",("cat1", "cat2", "cat3", "dog1", "dog2", "dog3"))
    # 可視化する画像を選択
    img_path = "dog_cat_img/{}.jpg".format(cats_dogs_var)

    # CNNの説明
    if st.sidebar.button("CNNとは？"):
        st.sidebar.markdown("畳み込み層(Convolution Layer)とプーリング層(Pooling Layer)から構成されるニューラルネットワークのことです。")
        
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    img_tensor /= 255.
    plt.imshow(img_tensor[0])

    # オリジナル画像を表示
    st.markdown("## オリジナル画像")
    st.pyplot()
    
    return img_tensor

# 特徴量マップを表示
def cnn_vis(layer_activation, layer_name, images_per_row, model):

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
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image

    # グリッド表示
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
    
    st.markdown("#### {}".format(layer_name))
    st.pyplot()
    

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def generate_pattern(layer_name, filter_index, model, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
      
def filter_vis(model, JPN_layer_name):
    layer_name_dict = {"畳み込み層 1":"conv2d", "畳み込み層 2":"conv2d_1", "畳み込み層 3":"conv2d_2", "畳み込み層 4":"conv2d_3"}
    layer_name = layer_name_dict[JPN_layer_name]
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype=int)

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 2), model, size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(10, 10))
    plt.imshow(results)
    st.pyplot()
    
    
                
def main():
    # streamlit上での警告を表示しないようにする
    st.set_option("deprecation.showPyplotGlobalUse", False)
    # 学習済みモデル「cats_and_dogs_small_1.h5」を用いる
    model = models.load_model("cats_and_dogs_small_1.h5")

    st.markdown("# 畳み込みニューラルネットワーク")
    
    # オリジナル画像を表示
    img_tensor = original_img()
    
    # 出力側の8つの層から出力を抽出
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # 特定の入力をもとに、これらの出力を返すモデルを作成
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # 5つのNumpy配列(層の活性化ごとに1つ)のリストを返す
    activations = activation_model.predict(img_tensor)

    # プロットの一部として使用する層の名前
    layer_names = ["畳み込み層 1", "プーリング層 1", "畳み込み層 2", "プーリング層 2", "畳み込み層 3", "プーリング層 3", "畳み込み層 4", "プーリング層 4"]
    
    images_per_row = 16
    
    # 畳み込み層 1 を表示
    cnn_vis(activations[0], layer_names[0], images_per_row, model)
    if st.button("{} のフィルター".format(layer_names[0])):
        filter_vis(model, layer_names[0])
    st.markdown(""" 畳み込み層は、入力画像をより特徴が強調されたものに変換します。
                特徴を検出する際に、画像の局所性を利用します。""")
    st.markdown("※局所性‥各ピクセルが近傍のピクセルと強い関連性を持っている性質のこと")
    
    # プーリング層 1 を表示
    cnn_vis(activations[1], layer_names[1], images_per_row, model)
    st.markdown("プーリング層は、画像を各領域に区切り、各領域を代表する値を抽出し、並べて新たな画像を生成します。")
    
    # 畳み込み層 2 を表示
    cnn_vis(activations[2], layer_names[2], images_per_row, model)
    if st.button("{} のフィルター".format(layer_names[2])):
        filter_vis(model, layer_names[2])
    
    # プーリング層 2 を表示
    cnn_vis(activations[3], layer_names[3], images_per_row, model)
    
    # 畳み込み層 3 を表示
    cnn_vis(activations[4], layer_names[4], images_per_row, model)
    if st.button("{} のフィルター".format(layer_names[4])):
        filter_vis(model, layer_names[4])
    
    # プーリング層 3 を表示
    cnn_vis(activations[5], layer_names[5], images_per_row, model)
    
    # 畳み込み層 4 を表示
    cnn_vis(activations[6], layer_names[6], images_per_row, model)
    if st.button("{} のフィルター".format(layer_names[6])):
        filter_vis(model, layer_names[6])
    
    # プーリング層 4 を表示
    cnn_vis(activations[7], layer_names[7], images_per_row, model)
    
if __name__ == "__main__":
    main()