from neo4j import GraphDatabase
import random
import json
from datetime import datetime

# ========== 配置 Neo4j 连接 ==========
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"  # ❗请替换为你的实际密码

# 初始化驱动
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# ========== 随机数据池 ==========
MOVIE_TITLES = [
    "无间道", "英雄", "疯狂的石头", "让子弹飞", "我不是药神", "流浪地球", "战狼", "哪吒之魔童降世",
    "红海行动", "唐人街探案", "你好，李焕英", "长津湖", "悬崖之上", "少年的你", "消失的她",
    "追风筝的人", "风声", "影", "一秒钟", "地久天长"
]

COMPANY_NAMES = [
    "华谊兄弟", "光线传媒", "博纳影业", "万达影视", "腾讯影业", "阿里影业",
    "上影集团", "中影集团", "环球影业", "华纳兄弟", "迪士尼中国", "派拉蒙中国"
]

GENRES = ["剧情", "动作", "喜剧", "爱情", "悬疑", "惊悚", "科幻", "战争", "文艺", "犯罪"]

# 简单的剧情模板
PLOT_TEMPLATES = [
    "讲述了一段关于{theme}的动人故事。",
    "在{setting}背景下，主人公经历了{challenge}。",
    "一部关于{theme}的深刻反思，引人深思。",
    "充满紧张气氛的{genre}片，结局出人意料。",
    "通过细腻的镜头语言，展现了{theme}的复杂性。"
]

THEMES = ["爱情", "背叛", "复仇", "成长", "家国情怀", "人性", "命运", "救赎"]
SETTINGS = ["民国时期", "现代都市", "边陲小镇", "未来世界", "战争年代"]
CHALLENGES = ["生死抉择", "身份危机", "情感纠葛", "道德困境"]


def random_plot():
    """生成一段随机剧情简介"""
    template = random.choice(PLOT_TEMPLATES)
    genre = random.choice(GENRES)
    theme = random.choice(THEMES)
    setting = random.choice(SETTINGS)
    challenge = random.choice(CHALLENGES)
    return template.format(genre=genre, theme=theme, setting=setting, challenge=challenge)


def random_release_date():
    """生成 2000 - 2023 年之间的随机日期（格式：YYYY年M月D日）"""
    year = random.randint(2000, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # 简化，避免 2 月问题
    return f"{year}年{month}月{day}日"


# ========== 写入函数 ==========
def create_random_movies_and_companies(num_movies=20, num_companies=10):
    """随机生成电影和公司，并建立关系"""
    with driver.session() as session:
        # 1. 创建公司（去重）
        created_companies = set()
        for _ in range(num_companies):
            name = random.choice(COMPANY_NAMES)
            if name in created_companies:
                continue
            session.run("""
                MERGE (c:Company {name: $name})
            """, name=name)
            created_companies.add(name)
        print(f"✅ 创建了 {len(created_companies)} 家公司")

        # 2. 创建电影并关联公司
        for _ in range(num_movies):
            title = random.choice(MOVIE_TITLES)
            # 避免重复电影名（简单处理）
            existing = session.run("MATCH (m:Movie {name: $name}) RETURN m", name=title).single()
            if existing:
                title = f"{title}_{random.randint(100, 999)}"  # 重命名

            plot = random_plot()
            release_date = random_release_date()

            # 随机选择 1-2 家公司
            company_name = random.choice(list(created_companies))

            # 创建电影并建立关系
            session.run("""
                CREATE (m:Movie {
                    name: $name,
                    plot: $plot,
                    release_date_str: $release_date_str
                })
                WITH m
                MATCH (c:Company {name: $company_name})
                CREATE (m)-[:PRODUCED_BY]->(c)
            """, name=title, plot=plot, release_date_str=release_date, company_name=company_name)

        print(f"✅ 创建了 {num_movies} 部电影，并随机关联到公司")

    print("🎉 数据写入完成！")


# ========== 主程序 ==========
if __name__ == "__main__":
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ 成功连接到 Neo4j")

        # 执行写入
        create_random_movies_and_companies(num_movies=15, num_companies=8)

    except Exception as e:
        print("❌ 连接失败:", e)
    finally:
        driver.close()