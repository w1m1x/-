import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('./raw_data.xlsx', sheet_name='Sheet1')

# 提取关系
all_relations = []
for i in range(len(df)):
    line = df.iloc[i, 1]
    # 如果 line 是 float 类型，则跳过
    if isinstance(line, float):
        continue

    # 处理 "贾府远房宗族" 特殊情况
    if line == "贾府远房宗族":
        relation = df.iloc[i, 0]
        relation += ",贾府,远房宗族"
        all_relations.append(relation.split(','))
        continue

    # 通过 '，' 和 '。' 分割
    line_segments = line.split('，')
    for segment in line_segments:
        sub_segments = segment.split('。')
        for sub_segment in sub_segments:
            relation = df.iloc[i, 0]
            if '的' in sub_segment:
                parts = sub_segment.split('的')
                relation += "," + parts[0] + "," + parts[1]
                all_relations.append(relation.split(','))
            elif '之' in sub_segment:
                parts = sub_segment.split('之')
                relation += "," + parts[0] + "," + parts[1]
                all_relations.append(relation.split(','))
            elif '长子' in sub_segment:
                relation += "," + sub_segment[:-2] + ",长子"
                all_relations.append(relation.split(','))
            elif '次子' in sub_segment:
                relation += "," + sub_segment[:-2] + ",次子"
                all_relations.append(relation.split(','))

# 过滤掉不符合格式的行
filtered_relations = [rel for rel in all_relations if len(rel) == 3]

# 进一步处理符号
clean_relations = []
for d in filtered_relations:
    if len(d) != 3:
        continue
    if d[2] == "一":
        continue

    entities = d[1].replace('、', ';').replace('；', ';').split(';')  # 处理'、'和'；'符号
    relations = d[2].replace('、', ';').replace('；', ';').split(';')  # 处理关系字段的'、'和'；'符号

    for entity in entities:
        for relation in relations:
            # 交换head和tail的位置
            clean_relations.append([entity.strip(), relation.strip(), d[0]])

# 将清理后的数据保存为 clean_relations.csv
df_clean_relations = pd.DataFrame(clean_relations, columns=['head', 'relation', 'tail'])
df_clean_relations.to_csv('clean_relations.csv', index=False, encoding='utf-8')

print("Cleaned relationships have been successfully saved as a CSV file.")
