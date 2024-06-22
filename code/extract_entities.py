import pandas as pd

# 读取 Excel 文件中的数据
df = pd.read_excel('./raw_data.xlsx', sheet_name='Sheet1')

# 提取实体
all_entities = []
for i in range(len(df)):
    # 获取第i行，第0列的数据
    value = df.iloc[i, 0]
    if pd.notna(value):  # 检查是否不是NaN
        all_entities.append(str(value))  # 将所有值转换为字符串并追加到列表

# 将实体写入到文本文件中
with open('entities.txt', 'w', encoding='utf-8') as f:
    for entity in all_entities:
        f.write(entity.strip() + '\n')
