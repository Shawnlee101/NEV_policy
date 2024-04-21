# -*- coding: 编码类型 -*-

import pandas as pd
import jieba
import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import find
from collections import Counter
import re
from datetime import datetime

NOW = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

# Jieduan = '技术探索期'
# Jieduan = '产业培育期'
Jieduan = '产业规律重视期'
# Jieduan = '发展质量提升期'


# 读取Excel文件中的政策文本
def read_policy_texts(file_path, sheet_name="Sheet1", text_column='文本'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    texts = df.loc[df["政策阶段"] == Jieduan, text_column].dropna().tolist()
    # texts = df[text_column].dropna().tolist()  # 假设M列为文本列
    return texts


# 使用jieba进行中文分词
def chinese_word_segmentation(texts: list) -> list:
    with open("stop_words.txt", "r", encoding="utf-8") as f:
        stop_words = [word.strip() for word in f]

    # 自定义词典
    jieba.load_userdict("custom_dict_0409_培育.txt")

    words = []
    pattern = re.compile(r'[\d\W]+')  # 匹配数字和符号的正则表达式
    for text in texts:
        # 去除数字和符号
        text = pattern.sub('', text)
        # 分词并过滤停用词
        words.extend([word for word in jieba.cut(text) if word not in stop_words])
    return words


def co_occurrence_matrix(words: list):
    with open("custom_dict_0409_培育.txt", "r", encoding="utf-8") as f:
        nev_words = [word.strip() for word in f]

    # 移除那些没有在words中出现的nev_words词汇
    nev_words = [word for word in nev_words if word in words]

    # 创建一个字典来存储每个关键词及其在words中的索引列表
    word_indices = {word: [index for index, w in enumerate(words) if w == word] for word in nev_words}

    # 初始化共现矩阵，矩阵大小为nev_words列表长度的平方
    co_occurrence_matrix = np.zeros((len(nev_words), len(nev_words)))

    # 设置词汇对中两个词的最大距离阈值
    max_distance = 6

    # 创建一个字典来存储每个关键词的共现次数
    co_occurrence_count = defaultdict(int)

    # 遍历专用词典中的词汇对
    for i, word1 in enumerate(nev_words):
        indices1 = word_indices[word1]
        for j, word2 in enumerate(nev_words[i + 1:], start=i + 1):
            indices2 = word_indices[word2]
            # 计算词汇对的共现次数
            for idx1 in indices1:
                for idx2 in indices2:
                    if abs(idx1 - idx2) <= max_distance:
                        co_occurrence_count[(word1, word2)] += 1
                        break  # 找到共现后，跳出内层循环

    # 填充共现矩阵
    for word_pair, count in co_occurrence_count.items():
        i, j = nev_words.index(word_pair[0]), nev_words.index(word_pair[1])
        co_occurrence_matrix[i][j] = count
        # 由于矩阵是对称的，我们只需要填充上三角部分
        if i != j:
            co_occurrence_matrix[j][i] = count

    # 将共现矩阵转换为DataFrame
    df = pd.DataFrame(co_occurrence_matrix, index=nev_words, columns=nev_words)
    # 保存共现矩阵到Excel文件
    # excel_filename = f"co_occurrence_matrix_{Jieduan}.xlsx"
    excel_filename = f"co_occurrence_matrix_{Jieduan}.xlsx"
    df.to_excel(excel_filename)

    # 打印共现矩阵
    # print("共现矩阵:")
    # print(co_occurrence_matrix)
    return co_occurrence_matrix, nev_words


def co_network(co_occurrence_matrix, nev_words):
    # 创建一个图
    G = nx.Graph()

    # 添加节点
    # for i, word in enumerate(nev_words):
    #     G.add_node(word, size=100)  # 默认节点大小设置为100

    # 添加边，这里边的权重表示共现次数
    for i, row in enumerate(co_occurrence_matrix):
        for j, value in enumerate(row):
            if value >= 1:  # 共现次数阈值
                G.add_edge(nev_words[i], nev_words[j], weight=value)

    # 计算点度中心度
    degree_centrality = nx.degree_centrality(G)

    # 调整节点大小，使其与点度中心度成正比
    for node in G.nodes():
        size = degree_centrality[node] * 1500  # 这里乘以一个系数来调整大小
        G.nodes[node]['size'] = size

    # 绘制图
    # 设置图像尺寸和分辨率
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)  # 例如，设置图像大小为10x8英寸，分辨率为100 DPI
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=[v['size'] for v in G.nodes.values()], node_color='#6495ED')  # 节点的大小，默认是600
    nx.draw_networkx_edges(G, pos, alpha=0.3,
                           edge_color="#a1a3a9")  # 边的透明度，范围是0（完全透明）到1（完全不透明）。在这个例子中，它被设置为0.3，意味着边将有一个较低的透明度，这可以使图看起来不那么拥挤
    edge_labels = {}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei', font_color='#3e4145')
    # 不显示坐标轴
    ax.axis('off')
    # 保存图片
    # plt.savefig(f"{NOW}.jpg", bbox_inches='tight', pad_inches=0.1)
    # 显示图
    plt.show()


# def less_word(words: list):
#     counter = Counter(words)
#     # 创建一个新列表，包含出现次数大于或等于50次的词
#     filtered_words = [word for word in words if counter[word] >= 200]
#     return filtered_words
#
#
# # 构建共现矩阵
# def build_cooccurrence_matrix(words_list):
#     word_to_index = defaultdict(int)
#     cooccurrence_counts = defaultdict(int)
#
#     # 初始化词汇索引
#     for word in words_list:
#         if word not in word_to_index:
#             word_to_index[word] = len(word_to_index)
#
#     # 计算词汇共现
#     for i in range(len(words_list)):
#         for j in range(i + 1, len(words_list)):
#             word1 = words_list[i]
#             word2 = words_list[j]
#             if word1 != word2:  # 避免同一个词与自己共现
#                 cooccurrence_counts[(word1, word2)] += 1
#
#     # 构建共现矩阵
#     rows, cols, data = [], [], []
#     for (word1, word2), count in cooccurrence_counts.items():
#         row_index = word_to_index[word1]
#         col_index = word_to_index[word2]
#         rows.append(row_index)
#         cols.append(col_index)
#         data.append(count)
#
#     matrix = coo_matrix((data, (rows, cols)), shape=(len(word_to_index), len(word_to_index)))
#     return matrix, word_to_index, words_list
#
#
# # 可视化共词网络
# def visualize_cooccurrence_network(matrix, word_to_index, words_list, weight_threshold):
#     # 创建一个 networkx 图
#     G = nx.Graph()
#
#     # 提取非零元素的行、列和值
#     row, col, data = find(matrix)
#
#     # 过滤边：只保留权重高于阈值的边
#     filtered_edges = [(row[i], col[i], data[i]) for i in range(len(data)) if data[i] > weight_threshold]
#
#     # 将行、列索引转换为对应的词汇
#     words = list(word_to_index.keys())
#     # edges = [(words[i], words[j], w) for i, j, w in zip(row, col, data)
#     edges = [(words[i], words[j], w) for i, j, w in filtered_edges]
#
#     # 将边添加到图中，并设置权重
#     G.add_weighted_edges_from(edges)
#
#     # 计算每个词语的频率
#     word_counts = Counter(words_list)
#
#     # 计算节点大小，这里假设words列表中的元素顺序与矩阵的行/列顺序对应
#     node_sizes = [word_counts[word] * 5 for word in G.nodes()]  # 乘以10是为了让节点大小更明显
#
#     # 使用 networkx 绘制网络图
#     # pos = nx.spring_layout(G)  # 使用 spring 布局算法
#     # pos = nx.circular_layout(G)  # 使用圆形布局算法
#     # pos = nx.shell_layout(G)  # 使用 shell 布局算法
#     pos = nx.kamada_kawai_layout(G)  # 使用 Kamada-Kawai 布局算法
#     # pos = nx.spectral_layout(G)  # 使用谱布局算法
#
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#a1a3a6')  # 节点的大小，默认是600
#     nx.draw_networkx_edges(G, pos, alpha=0.3,
#                            edge_color="#a1a3a9")  # 边的透明度，范围是0（完全透明）到1（完全不透明）。在这个例子中，它被设置为0.3，意味着边将有一个较低的透明度，这可以使图看起来不那么拥挤
#     # edge_labels = {(u, v): d for u, v, d in G.edges(data=True)}
#     edge_labels = {}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei', font_color='#3e4145')
#     plt.axis('off')  # 不显示坐标轴
#     plt.savefig(f"{NOW}.jpg")
#     plt.show()
#
#     # plt.close()


# 主函数
# def main(file_path):
#     texts = read_policy_texts(file_path)
#     words = chinese_word_segmentation(texts)
#
#     # 示例使用
#     # words_list = ['低能耗', '新能源', '人才', '培养', '引进', '产研一体化', '新能源', '工匠精神', '汽车', '人才', '人才']
#     # matrix, word_to_index, words_list = build_cooccurrence_matrix(words_list)
#
#     # 打印结果
#     # print("共现矩阵:")
#     # print(matrix)
#     # print("\n词汇索引:")
#     # print(word_to_index)
#
#     # words = less_word(words)
#     matrix, word_to_index, words_list = build_cooccurrence_matrix(words)
#     visualize_cooccurrence_network(matrix, word_to_index, words_list, weight_threshold=80000)


# 调用主函数
if __name__ == "__main__":
    # file_path = '230302/all_data.xlsx'  # 替换为您的Excel文件路径
    # main(file_path)
    # words = [
    #     ' '
    #     ]
    words = read_policy_texts('230302/all_data.xlsx')
    words = chinese_word_segmentation(words)
    co_occurrence_matrix, nev_words = co_occurrence_matrix(words=words)
    co_network(co_occurrence_matrix, nev_words)
    print(words)
