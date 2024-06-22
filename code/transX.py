import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models.predict import predict_triples_df
from pykeen.evaluation import RankBasedEvaluator

# Step 1: 从CSV文件读取数据，忽略错误行
df = pd.read_csv('clean_relations.csv', on_bad_lines='skip')

# 确保所有列都是字符串类型，并处理缺失值
df = df.fillna('').astype(str)

# 检查读取的 DataFrame
print(df.head())

# Step 2: 将数据转换为知识图谱的三元组工厂
triples_factory = TriplesFactory.from_labeled_triples(
    df[['head', 'relation', 'tail']].values
)

# Step 3: 调整拆分比例
training, testing, validation = triples_factory.split([0.9, 0.05, 0.05])

# Step 4: 使用 PyKEEN 的 pipeline 训练 TransE 模型
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    training_kwargs=dict(num_epochs=200),
    random_seed=42,
    device='cpu'
)

# 获取训练好的模型
model = result.model


# Step 5: 计算 Mean Rank 和 Hits@10

def evaluate_model_mean_rank_hits_at_k(model, testing, k=10):
    evaluator = RankBasedEvaluator()
    evaluation_results = evaluator.evaluate(model, mapped_triples=testing.mapped_triples)

    mean_rank = evaluation_results.get_metric('mean_rank')
    hits_at_k = evaluation_results.get_metric(f'hits@{k}')

    print(f"Mean Rank: {mean_rank}")
    print(f"Hits@{k}: {hits_at_k}")

    return mean_rank, hits_at_k


# 评估模型并打印 Mean Rank 和 Hits@10
mean_rank, hits_at_10 = evaluate_model_mean_rank_hits_at_k(model, testing, k=10)


# Step 6: 预测所有测试集三元组的得分并保存到CSV文件

def predict_and_save_all_triples_scores(model, testing, triples_factory, output_file):
    # 获取所有实体和关系的映射
    id_to_entity = {v: k for k, v in triples_factory.entity_to_id.items()}
    id_to_relation = {v: k for k, v in triples_factory.relation_to_id.items()}

    # 将测试集中的三元组转化为实体和关系标签
    triples = [(id_to_entity[row[0]], id_to_relation[row[1]], id_to_entity[row[2]]) for row in
               testing.mapped_triples.numpy()]

    # 使用 predict_triples_df 函数对所有测试集中的三元组进行评分
    df_scores = predict_triples_df(model=model, triples=triples, triples_factory=triples_factory)

    # 保存到CSV文件
    df_scores.to_csv(output_file, index=False, encoding='utf-8')
    print(f"All test set scores have been saved to {output_file}")


# 保存所有测试集的结果
output_file_scores = 'all_test_set_scores.csv'
predict_and_save_all_triples_scores(model, testing, triples_factory, output_file_scores)


# Step 7: 根据固定头实体和关系，预测尾实体的打分排序

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
head = '贾政'
relation = '次子'

# 获取预测的尾实体
predicted_tails = predict_tail_entities(model, triples_factory, head, relation, top_k=10)

print(f"Predicted Tail Entities for ({head}, {relation}):")
print(predicted_tails)

# 保存预测的尾实体结果到 CSV 文件中
output_file_predictions = f'predicted_tails_for_{head}_{relation}.csv'
predicted_tails.to_csv(output_file_predictions, index=False, encoding='utf-8')
print(f"Predicted tail entities have been saved to {output_file_predictions}")


