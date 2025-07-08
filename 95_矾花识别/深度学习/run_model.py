import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # 用于加载Scaler
import matplotlib.pyplot as plt

# 设置全局字体为中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_and_evaluate_model(data_path, model_load_path, scaler_load_path):
    # 第0步: 读取数据集
    data = pd.read_excel(data_path)  # 请确保提供正确的数据路径

    # 第1步: 准备数据 (提取特征和目标变量)
    X = data.drop(columns=['除浊率'])  # 除去目标变量"除浊率"
    y = data['除浊率']  # 目标变量

    # 第2步: 加载并应用训练时使用的Scaler
    scaler = joblib.load(scaler_load_path)
    X_scaled = scaler.transform(X)  # 注意这里使用transform而不是fit_transform

    # 将数据重塑为CNN输入格式
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # 第3步: 加载模型（无需指定自定义对象）
    model = tf.keras.models.load_model(model_load_path)

    # 第4步: 使用模型进行预测
    y_pred = model.predict(X_reshaped)

    # 第5步: 评估模型性能
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"测试集均方误差（MSE）: {mse}")
    print(f"测试集平均绝对误差（MAE）: {mae}")

    # 可视化预测值与实际值的对比
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y)), y, label="真实值", marker='o')
    plt.plot(np.arange(len(y_pred)), y_pred, label="预测值", marker='x')
    plt.title('真实值与预测值对比')
    plt.xlabel('样本索引')
    plt.ylabel('除浊率')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 示例用法，更新模型加载路径
    data_path = r"D:\大学生活\大创\基于卷积神经网络（CNN）的净水厂混凝除浊图像识别与调控策略研究\中期\训练集.xlsx"  # 请替换为你的测试集路径
    model_load_path = './模型/model_01.keras'  # 不再使用 '.h5' 扩展名
    scaler_load_path = './scaler/scaler_01.save'
    load_and_evaluate_model(data_path, model_load_path, scaler_load_path)
