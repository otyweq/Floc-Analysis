import matplotlib.pyplot as plt
import pandas as pd

# Set global font to SimHei (黑体)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Data from the user's input
data = {
    '实验': ['实验1', '实验2', '实验3', '实验4', '实验5', '实验6', '实验7', '实验8', '实验9'],
    '加药量': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    '絮凝剂种类': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    '混凝GT值': [1, 2, 3, 3, 3, 2, 2, 1, 1],
    'PH': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    '实验结果': [30.19, 14.21, 1.46, 30.45, 53.41, 8.67, 17.4, 76.78, 59.41]
}

df_numeric = pd.DataFrame(data)

# Mapping the description details from the second image
description = {
    '加药量': {1: '25mg', 2: '30mg', 3: '35mg'},
    '絮凝剂种类': {1: '聚合氯化铝', 2: '三氯化铁', 3: '硫酸铝'},
    '混凝GT值': {1: '24351(80)', 2: '26670(85)', 3: '29057(90)'},
    'PH': {1: 6, 2: 7, 3: 8}
}

# Replace the numeric values with descriptions
df_numeric['加药量'] = df_numeric['加药量'].map(description['加药量'])
df_numeric['絮凝剂种类'] = df_numeric['絮凝剂种类'].map(description['絮凝剂种类'])
df_numeric['混凝GT值'] = df_numeric['混凝GT值'].map(description['混凝GT值'])
df_numeric['PH'] = df_numeric['PH'].map(description['PH'])

# Create the bar charts and show the values on the bars
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
factor_columns = ['加药量', '絮凝剂种类', '混凝GT值', 'PH']

for i, (factor, ax) in enumerate(zip(factor_columns, axes.flatten())):
    bar_plot = df_numeric.groupby(factor)['实验结果'].mean().plot(kind='bar', ax=ax, color=['r', 'g', 'b'])
    ax.set_title(f'{factor} vs 除浊率')
    ax.set_ylabel('除浊率 (%)')
    ax.set_xlabel(factor)

    # Displaying the values on the bars
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.2f}%',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.tight_layout()
plt.show()
