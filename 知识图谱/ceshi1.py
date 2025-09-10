import jieba.posseg as pseg


def extract_entities(text):
    """提取人名、地名、机构名"""
    entities = {
        "person": [],  # 人名 (nr)
        "place": [],  # 地名 (ns)
        "organization": []  # 机构名 (nt)
    }

    for word, flag in pseg.cut(text):
        if flag == 'nr':  # nr = 人名
            entities["person"].append(word)
        elif flag == 'ns':  # ns = 地名
            entities["place"].append(word)
        elif flag == 'nt':  # nt = 机构名
            entities["organization"].append(word)

    return entities


# 测试文本
text = "周星驰导演了电影《功夫》，他在阿里影业工作，位于杭州。"

# 提取实体
result = extract_entities(text)

# 打印结果
print("🔍 实体识别结果：")
print(f"人名: {result['person']}")
print(f"地名: {result['place']}")
print(f"机构: {result['organization']}")