import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models.predict import predict_triples_df
from pykeen.evaluation import RankBasedEvaluator
from sklearn.model_selection import KFold

# Step 1: 从CSV文件读取数据，忽略错误行
df = pd.read_csv('clean_relations.csv', on_bad_lines='skip')

# 确保所有列都是字符串类型，并处理缺失值
df = df.fillna('').astype(str)

# 检查读取的 DataFrame
print(df.head())

# Step 2: 将数据转换为知识图谱的三元组工厂
triples = df[['head', 'relation', 'tail']].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mean_ranks = []
hits_at_10 = []

for train_index, test_index in kf.split(triples):
    train_triples, test_triples = triples[train_index], triples[test_index]

    # 创建训练和测试三元组工厂
    train_factory = TriplesFactory.from_labeled_triples(train_triples)
    test_factory = TriplesFactory.from_labeled_triples(test_triples)

    # 使用 PyKEEN 的 pipeline 训练 TransE 模型
    result = pipeline(
        training=train_factory,
        testing=test_factory,
        model='TransE',
        training_kwargs=dict(num_epochs=100),
        random_seed=42,
        device='cpu'  # 或 'cuda' 来使用 GPU
    )

    # 获取训练好的模型
    model = result.model

    # 评估模型
    evaluator = RankBasedEvaluator()
    evaluation_results = evaluator.evaluate(model, mapped_triples=test_factory.mapped_triples)

    mean_rank = evaluation_results.get_metric('mean_rank')
    hits_10 = evaluation_results.get_metric('hits@10')

    mean_ranks.append(mean_rank)
    hits_at_10.append(hits_10)

    print(f"Fold {len(mean_ranks)} - Mean Rank: {mean_rank}, Hits@10: {hits_10}")

# 计算并打印平均 Mean Rank 和 Hits@10
mean_rank_avg = sum(mean_ranks) / len(mean_ranks)
hits_at_10_avg = sum(hits_at_10) / len(hits_at_10)

print(f"Average Mean Rank: {mean_rank_avg}")
print(f"Average Hits@10: {hits_at_10_avg}")

# 使用整个数据集重新训练模型并进行预测
final_result = pipeline(
    training=TriplesFactory.from_labeled_triples(triples),
    model='TransE',
    training_kwargs=dict(num_epochs=100),
    random_seed=42,
    device='cpu'  # 或 'cuda' 来使用 GPU
)

model = final_result.model


# 预测所有三元组的得分并保存到CSV文件
def predict_and_save_all_triples_scores(model, triples_factory, output_file):
    # 获取所有实体和关系的映射
    id_to_entity = {v: k for k, v in triples_factory.entity_to_id.items()}
    id_to_relation = {v: k for k, v in triples_factory.relation_to_id.items()}

    # 获取所有三元组
    triples = [(id_to_entity[row[0]], id_to_relation[row[1]], id_to_entity[row[2]]) for row in
               triples_factory.mapped_triples.numpy()]

    # 使用 predict_triples_df 函数对所有三元组进行评分
    df_scores = predict_triples_df(model=model, triples=triples, triples_factory=triples_factory)

    # 保存到CSV文件
    df_scores.to_csv(output_file, index=False, encoding='utf-8')
    print(f"All triples scores have been saved to {output_file}")


# 保存所有三元组的结果
output_file_scores = 'all_triples_scores.csv'
predict_and_save_all_triples_scores(model, TriplesFactory.from_labeled_triples(triples), output_file_scores)


# 根据固定头实体和关系，预测尾实体的打分排序
def predict_tail_entities(model, triples_factory, head, relation, top_k=10):
    # 构造需要预测的头实体和关系，所有可能的尾实体
    tail_candidates = list(triples_factory.entity_to_id.keys())

    # 构造三元组 (head, relation, tail_candidate) 形式
    triples_to_predict = [(head, relation, tail) for tail in tail_candidates]

    # 使用 predict_triples_df 方法预测尾实体的得分
    df = predict_triples_df(
        model=model,
        triples=triples_to_predict,
        triples_factory=triples_factory
    )

    # 根据得分降序排序，并选择前 top_k 个
    top_predictions = df.sort_values(by='score', ascending=False).head(top_k)

    return top_predictions


# 指定要预测的头实体和关系
head = '贾宝玉'
relation = '次子'

# 获取预测的尾实体
predicted_tails = predict_tail_entities(model, TriplesFactory.from_labeled_triples(triples), head, relation, top_k=10)

print(f"Predicted Tail Entities for ({head}, {relation}):")
print(predicted_tails)

# 保存预测的尾实体结果到 CSV 文件中
output_file_predictions = f'predicted_tails_for_{head}_{relation}.csv'
predicted_tails.to_csv(output_file_predictions, index=False, encoding='utf-8')
print(f"Predicted tail entities have been saved to {output_file_predictions}")