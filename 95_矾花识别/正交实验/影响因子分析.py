import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 数据：用户实验的数值形式，其中'加药量'、'絮凝剂种类'、'混凝GT值'和'PH'为自变量，'实验结果'为因变量
data_numeric = {
    '加药量': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    '絮凝剂种类': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    '混凝GT值': [1, 2, 3, 3, 3, 2, 2, 1, 1],
    'PH': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    '实验结果': [30.19, 14.21, 1.46, 30.45, 53.41, 8.67, 17.4, 76.78, 59.41]
}

# 将数据转换为DataFrame格式，方便进行后续操作
df_numeric_analysis = pd.DataFrame(data_numeric)

# 提取自变量（即“加药量”、“絮凝剂种类”、“混凝GT值”、“PH”）和因变量（即“实验结果”）
X = df_numeric_analysis[['加药量', '絮凝剂种类', '混凝GT值', 'PH']]
y = df_numeric_analysis['实验结果']

# 创建多项式特征（degree=2 表示考虑二次项和交互项，include_bias=False 说明不加入常数项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # X_poly是扩展后的自变量矩阵

# 创建线性回归模型
model = LinearRegression()
model.fit(X_poly, y)  # 拟合模型

# 获取回归模型的系数和截距
coefficients = model.coef_  # 回归系数
intercept = model.intercept_  # 截距

# 输出回归模型的结果（截距和系数）
print("回归方程的截距（常数项）：", intercept)
print("回归方程的系数：", coefficients)

# 可选：查看生成的多项式特征名称，帮助理解每个系数对应的项
print("多项式特征的名称：", poly.get_feature_names_out(['加药量', '絮凝剂种类', '混凝GT值', 'PH']))


'''
PH: 26.1319（正向影响）
混凝GT值 PH 交互项: -19.1102（负向影响）
混凝GT值: -17.4556（负向影响）
加药量 PH 交互项: 13.7668（正向影响）
絮凝剂种类 PH 交互项: 13.7668（正向影响）
混凝GT值^2: 8.3865（正向影响）
加药量 混凝GT值 交互项: 6.7901（正向影响）
絮凝剂种类 混凝GT值 交互项: 6.7901（正向影响）
加药量^2: -6.7839（负向影响）
加药量 絮凝剂种类 交互项: -6.7839（负向影响）
絮凝剂种类^2: -6.7839（负向影响）
絮凝剂种类: -5.8964（负向影响）
加药量: -5.8964（负向影响）
PH^2: -5.7317（负向影响）
'''