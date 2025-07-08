import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # 用于保存和加载Scaler
import matplotlib.pyplot as plt

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def train_and_save_model(data_path, model_save_path, scaler_save_path):
    # 第0步: 读取数据集
    data = pd.read_excel(data_path)  # 请确保提供正确的数据路径

    # 第1步: 准备数据 (提取特征和目标变量)
    X = data.drop(columns=['除浊率'])  # 除去目标变量"除浊率"，其余作为特征
    y = data['除浊率']  # 目标变量

    # 第2步: 对特征进行标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 对特征进行归一化

    # 保存Scaler以便在测试时使用
    joblib.dump(scaler, scaler_save_path)

    # 将数据重塑为CNN输入格式 (假设1D结构，我们为“通道”添加另一维度)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # 第3步: 将数据划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42)

    # 第4步: 构建CNN模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu',
                               input_shape=(X_train.shape[1], 1)),  # 卷积层
        tf.keras.layers.MaxPooling1D(pool_size=2),  # 池化层
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),  # 第二个卷积层
        tf.keras.layers.Flatten(),  # 展平
        tf.keras.layers.Dense(64, activation='relu'),  # 全连接层
        tf.keras.layers.Dense(1)  # 输出层用于回归
    ])

    # 第5步: 编译模型，使用明确的损失函数
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # 第6步: 训练模型
    history = model.fit(X_train, y_train, epochs=50,
                        validation_data=(X_val, y_val), verbose=1)

    # 第7步: 保存模型（使用 SavedModel 格式）
    model.save(model_save_path)

    # 可选：保存训练历史
    # np.save('training_history.npy', history.history)

    # 绘制训练过程中损失的变化曲线
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练损失与验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 示例用法，更新模型保存路径
    data_path = r"D:\大学生活\大创\基于卷积神经网络（CNN）的净水厂混凝除浊图像识别与调控策略研究\中期\训练集.xlsx"  # 请替换为你的训练集路径
    model_save_path = './模型/model_01.keras'  # 不再使用 '.h5' 扩展名
    scaler_save_path = './scaler/scaler_01.save'
    train_and_save_model(data_path, model_save_path, scaler_save_path)
