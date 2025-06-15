
# def double_net():
#     # 输入：样本对 (x1, x2) 和标签 y_pair（0/1）
#     f1 = encoder(x1)  # 共享编码器
#     f2 = encoder(x2)
#     distance = torch.norm(f1 - f2, p=2)  # 欧氏距离
#     prob = sigmoid(distance)  # 相似度概率
#     loss = binary_cross_entropy(prob, y_pair)
#
# def primary_net():
#     # 输入：支撑集 (support_set) 和查询样本 (query)
#     class_prototypes = []
#     for k in classes:
#         samples_k = support_set[label == k]
#         features_k = encoder(samples_k)  # 编码支撑集样本
#         prototype_k = mean(features_k, dim=0)  # 计算类原型
#         class_prototypes.append(prototype_k)
#
#     query_feature = encoder(query)  # 编码查询样本
#     distances = [norm(query_feature - c_k) for c_k in class_prototypes]
#     probs = softmax(-torch.stack(distances))  # 负距离Softmax
#     loss = cross_entropy(probs, query_label)

"""
孪生网络是样本级比对的基石，适合二元匹配任务，但分类效率低；

原型网络是类别中心化的典范，通过类原型实现高效少样本分类，但对类内分布敏感。
两者共同推动了小样本学习的发展，实际应用中常结合使用（如用孪生网络预训练特征编码器，再用原型网络分类）。
"""

