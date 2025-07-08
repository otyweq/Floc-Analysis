import pandas as pd

# 用户提供的数据
data = {
    'Dosage': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'Coagulant_Type': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'Mixing_GT_Value': [1, 2, 3, 3, 3, 2, 2, 1, 1],
    'pH': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'Result': [30.19, 14.21, 1.46, 30.45, 53.41, 8.67, 17.4, 76.78, 59.41]
}

df = pd.DataFrame(data)
print(df)

# 计算每个因素在各水平下的平均值
factors = ['Dosage', 'Coagulant_Type', 'Mixing_GT_Value', 'pH']
range_analysis = {}

for factor in factors:
    means = df.groupby(factor)['Result'].mean()
    range_value = means.max() - means.min()
    range_analysis[factor] = range_value

# 转换为DataFrame以便查看
range_df = pd.DataFrame(list(range_analysis.items()), columns=['Factor', 'Range'])
range_df = range_df.sort_values(by='Range', ascending=False)
print(range_df)

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 构建回归模型（仅考虑主效应）
formula = 'Result ~ C(Dosage) + C(Coagulant_Type) + C(Mixing_GT_Value) + C(pH)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
