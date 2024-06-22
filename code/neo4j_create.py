import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher

# Step 1: 读取 CSV 文件
df = pd.read_csv('clean_relations.csv')

# Step 2: 设置 Neo4j 数据库连接
# 请根据你的数据库设置修改以下 URL、用户名和密码
uri = "bolt://localhost:7687"  # Neo4j 服务的地址
username = "neo4j"  # Neo4j 用户名
password = "***"  # Neo4j 密码

# 创建 Graph 对象
graph = Graph(uri, auth=(username, password))

# Step 3: 清理现有数据（可选）
# 你可以选择清空数据库，也可以保留现有数据
graph.delete_all()

# Step 4: 创建节点和关系
for index, row in df.iterrows():
    head = str(row['head']).strip()  # 确保转换为字符串并去除首尾空格
    tail = str(row['tail']).strip()  # 确保转换为字符串并去除首尾空格
    relation = str(row['relation']).strip()  # 确保转换为字符串并去除首尾空格

    # 处理空值或 NaN 的情况
    if pd.isna(head) or pd.isna(tail) or pd.isna(relation):
        continue

    # 清理关系中的空白字符并转换为适合 Cypher 查询的格式
    relation = relation.replace(" ", "_").upper()

    # 使用 NodeMatcher 查找或创建节点
    try:
        head_node = graph.nodes.match("Person", name=head).first()
        if not head_node:
            head_node = Node("Person", name=head)
            graph.create(head_node)

        tail_node = graph.nodes.match("Person", name=tail).first()
        if not tail_node:
            tail_node = Node("Person", name=tail)
            graph.create(tail_node)

        # 检查关系是否已经存在，避免重复创建
        matcher = NodeMatcher(graph)
        existing_relation = graph.match((head_node, tail_node), relation).first()
        if not existing_relation:
            rel = Relationship(head_node, relation, tail_node)
            graph.create(rel)

    except Exception as e:
        print(f"An error occurred: {e}")

# Step 5: 检查结果（打印前25个关系）
try:
    results = graph.run("MATCH (h)-[r]->(t) RETURN h.name, type(r), t.name LIMIT 25").data()
    for record in results:
        print(f"{record['h.name']} - [{record['type(r)']}] -> {record['t.name']}")
except Exception as e:
    print(f"An error occurred while retrieving relationships: {e}")

print("Data has been successfully imported to Neo4j and relationships created.")
