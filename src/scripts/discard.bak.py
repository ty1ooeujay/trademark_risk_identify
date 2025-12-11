import json
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

def remove_numbers_from_trademarks(input_file, output_file=None):
    """
    去除 trademarks.txt 文件中每行的序号
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    # 如果未指定输出文件，使用输入文件名
    if output_file is None:
        output_file = input_file
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行：去除开头的数字序号和点号空格
    processed_lines = []
    for line in lines:
        # 使用正则表达式去除开头的数字、点号和空格
        # 匹配模式：开头的数字 + 点号 + 空格
        cleaned_line = re.sub(r'^\d+\.\s+', '', line)
        processed_lines.append(cleaned_line)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"处理完成！共处理 {len(processed_lines)} 行")
    print(f"输出文件：{output_file}")


def trademark_replace_simchars():
    """
    将 trademarks.txt 文件中的字符替换为相似字符对
    """
    def replace_loop(trademark, char, sim_chars):
        results = []
        for sim_char in sim_chars:
            sim_trademark = trademark.replace(char, sim_char)
            result = {
                "trademark1": trademark,
                "trademark2": sim_trademark,
                }
            results.append(result)
        return results

    def after_processor(results):
        new_reults = []
        for result in results:
            result["trademark1"] = result["trademark1"].strip()
            result["trademark2"] = result["trademark2"].strip()
            if result["trademark1"] != result["trademark2"]:
                replace_num = 0
                for i in range(len(result["trademark1"])):
                    if result["trademark1"][i] != result["trademark2"][i]:
                        replace_num += 1

                result["sim"] = 1 -replace_num / len(result["trademark1"])
                new_reults.append(result)
        return new_reults

    results = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sim_char_filepath = os.path.join(project_root,"data","similar_chars.json")
    trademark_filepath = os.path.join(project_root,"data", "trademarks.txt")
    output_filepath = os.path.join(project_root,"data", "similar_trademark_pairs.json")
    with open(sim_char_filepath,"r", encoding='utf-8') as file1:
        sim_chars_list = json.load(file1)
    with open(trademark_filepath,"r", encoding='utf-8') as file2:
        trademarks = file2.readlines()
    for trademark in trademarks:
        sim_trademark_cluster = []
        for index, char in enumerate(trademark):
            for sim_chars in sim_chars_list:
                if char in sim_chars:
                    if index != 0:
                        for sim_trademark in sim_trademark_cluster:
                            results.extend(replace_loop(sim_trademark, char, sim_chars))
                    results.extend(replace_loop(trademark, char, sim_chars))
    results = after_processor(results)
    with open(output_filepath, "w", encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False,indent=2)

def sim_char_cluster():
    """
    将相似字符对转换为聚类数组，所有相互关联的相似字符归入同一数组。
    使用图连通分量算法实现。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_file = os.path.join(project_root, 'data', 'similar_chars_pairs.json')
    output_file = os.path.join(project_root, 'data', 'similar_chars_clusters.json')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        similar_chars_pairs = json.load(f)
    
    # 构建无向图（邻接表）
    graph = {}
    for pair in similar_chars_pairs:
        char1 = pair['char1']
        char2 = pair['char2']
        
        # 初始化节点
        if char1 not in graph:
            graph[char1] = set()
        if char2 not in graph:
            graph[char2] = set()
        
        # 添加边（无向图）
        graph[char1].add(char2)
        graph[char2].add(char1)
    
    # 使用 DFS 查找所有连通分量
    visited = set()
    sim_char_clusters = []
    
    def dfs(node, cluster):
        """深度优先搜索，找到连通分量中的所有节点"""
        visited.add(node)
        cluster.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, cluster)
    
    # 遍历所有节点，找到所有连通分量
    for char in graph:
        if char not in visited:
            cluster = []
            dfs(char, cluster)
            # 对每个聚类进行排序，保持一致性
            cluster.sort()
            sim_char_clusters.append(cluster)
    
    # 对聚类数组进行排序（按第一个字符排序）
    sim_char_clusters.sort(key=lambda x: x[0] if x else '')
    
    # 写入输出文件，格式与 similar_chars.json 一致（每个数组在一行）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for cluster in sim_char_clusters:
            # 将数组格式化为一行，格式：["char1","char2","char3"],
            cluster_str = json.dumps(cluster, ensure_ascii=False)
            f.write(f'    {cluster_str},\n')
        f.write(']')
    
    print(f"聚类完成！共生成 {len(sim_char_clusters)} 个聚类")
    print(f"输出文件：{output_file}")
    
    return sim_char_clusters

if __name__ == '__main__':    
    # 执行处理
    trademark_replace_simchars()

