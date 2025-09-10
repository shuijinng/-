# ========== 配置 ==========
from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase
import jieba.posseg as pseg

STATIC_FOLDER = "static"
TEMPLATES_FOLDER = "templates"

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"  # ❗替换为你的密码

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATES_FOLDER)

# ========== 连接 Neo4j ==========
try:
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ 成功连接到 Neo4j 数据库")
except Exception as e:
    print("❌ 无法连接到 Neo4j:", e)
    driver = None


# ========== 1. 实体提取函数 ==========
def extract_entities_from_text(text):
    entities = {"person": [], "place": [], "organization": []}
    for word, flag in pseg.cut(text):
        if flag == 'nr':
            entities["person"].append(word)
        elif flag == 'ns':
            entities["place"].append(word)
        elif flag == 'nt':
            entities["organization"].append(word)
    return entities


# ========== 2. 写入 Neo4j 的函数（提前定义！）==========
def write_entities_to_neo4j(session, entities):
    """
    将提取出的实体写入 Neo4j，并尽量保持与现有 schema 一致
    """
    for name in entities["person"]:
        session.write_transaction(
            lambda tx, n: tx.run("MERGE (p:Person {name: $name})", name=n), n=name
        )

    for name in entities["place"]:
        # 使用更通用的 Label，或者根据业务决定是否用 City/Country
        session.write_transaction(
            lambda tx, n: tx.run("MERGE (l:Place {name: $name})", name=n), n=name
        )

    for name in entities["organization"]:
        # 注意：原图谱可能是 Company，这里你可以选择合并或新建 Organization
        session.write_transaction(
            lambda tx, n: tx.run("MERGE (o:Company {name: $name}) RETURN o", name=n), n=name
        )


# ========== 3. 查询子图函数 ==========
def query_subgraph_by_name(session, name):
    result = session.run("""
        MATCH (seed)
        WHERE toLower(seed.name) CONTAINS toLower($name)
          AND seed.name IS NOT NULL
        WITH seed
        LIMIT 3

        MATCH path1 = (seed)-[r1]-(neighbor)
        WHERE neighbor.name IS NOT NULL
        LIMIT 15
        OPTIONAL MATCH (neighbor)-[r2]->(neighbor2)
        WHERE 
          (neighbor2:Genre OR neighbor2:Country OR neighbor2:Year OR neighbor2:Language)
          OR neighbor2.name IS NOT NULL

        RETURN
          seed,
          collect(DISTINCT {rel: r1, node: neighbor}) as direct_neighbors,
          collect(DISTINCT {rel: r2, node: neighbor2}) as second_hop
        LIMIT 30
    """, name=name)

    records = list(result)
    if not records:
        return f"🔍 未找到与“{name}”相关的信息。", None

    nodes, edges = [], []
    node_id_map = {}
    current_id = 0

    def add_node(node, group_hint=None):
        nonlocal current_id
        if not node:
            return None
        key = str(node.element_id)
        if key in node_id_map:
            return node_id_map[key]

        labels = node.labels

        if "Movie" in labels:
            group = "movie"
            prefix = "🎬 电影"
            extra = f"<br>年份: {node.get('year', '未知')}<br>评分: {node.get('rating', '未知')}"
        elif "Person" in labels:
            group = "person"
            prefix = "👤 人物"
            extra = f"<br>出生: {node.get('born', '未知')}"
        elif "Company" in labels or "Organization" in labels:
            group = "company"
            prefix = "🏢 公司/组织"
            extra = f"<br>国家: {node.get('country', '未知')}"
        elif "Location" in labels:  # 👈 新增：支持 Location
            group = "other"  # 你可以设为 'location'，但确保前端有对应颜色
            prefix = "🌍 地点"
            extra = ""
        elif "Genre" in labels:
            group = "other"
            prefix = "🏷️ 类型"
            extra = ""
        elif "Year" in labels:
            group = "other"
            prefix = "📅 年份"
            extra = ""
        else:
            group = group_hint or "other"
            prefix = "❓ 未知"
            extra = ""

        node_id_map[key] = current_id
        nodes.append({
            "id": current_id,
            "label": node.get("name") or "未知",
            "group": group,
            "title": f"{prefix}<br>名称: {node.get('name') or '未知'}{extra}"
        })
        current_id += 1
        return current_id - 1

    for record in records:
        seed = record["seed"]
        seed_id = add_node(seed)

        for item in record["direct_neighbors"]:
            rel = item["rel"]
            neighbor = item["node"]
            neighbor_id = add_node(neighbor)
            if neighbor_id is not None:
                edges.append({
                    "from": seed_id,
                    "to": neighbor_id,
                    "label": type(rel).__name__
                })

        for item in record["second_hop"]:
            rel = item["rel"]
            neighbor2 = item["node"]
            if neighbor2:
                neighbor2_id = add_node(neighbor2, group_hint="other")
                edges.append({
                    "from": neighbor_id,
                    "to": neighbor2_id,
                    "label": type(rel).__name__,
                    "color": {"color": "#aaaaaa"}
                })

    graph = {"nodes": nodes, "edges": edges}
    answer = f"🔍 找到与“{name}”相关的 {len(nodes)} 个实体和 {len(edges)} 条关系。"
    return answer, graph


# ========== 4. API 接口 ==========
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract_and_query", methods=["POST"])
def extract_and_query():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "success": False,
            "error": "❌ 请输入一段话。",
            "entities": None
        })

    # Step 1: 提取实体
    entities = extract_entities_from_text(text)
    all_extracted = entities["person"] + entities["place"] + entities["organization"]

    if not all_extracted:
        return jsonify({
            "success": False,
            "error": "⚠️ 未识别出有效实体（人名/地名/机构名）。",
            "entities": entities
        })

    # Step 2: 写入 Neo4j
    try:
        with driver.session() as session:
            write_entities_to_neo4j(session, entities)
        message = f"成功保存：人名 {len(entities['person'])}，地名 {len(entities['place'])}，机构名 {len(entities['organization'])}"
    except Exception as e:
        print("❌ 写入 Neo4j 失败:", e)
        return jsonify({
            "success": False,
            "error": f"写入数据库失败: {str(e)}",
            "entities": entities
        })

    return jsonify({
        "success": True,
        "text": text,
        "entities": entities,
        "message": message
    })


@app.route("/ask", methods=["POST"])
def ask():
    if not driver:
        return jsonify({"answer": "❌ 数据库连接失败", "graph": None}), 500

    data = request.get_json()
    name = data.get("question", "").strip()

    if not name:
        return jsonify({"answer": "❌ 请输入一个名字。", "graph": None})

    try:
        with driver.session() as session:
            answer, graph = query_subgraph_by_name(session, name)
    except Exception as e:
        print("❌ 查询出错:", e)
        return jsonify({"answer": f"❌ 查询失败：{str(e)}", "graph": None}), 500

    return jsonify({"answer": answer, "graph": graph})


@app.route("/api/entities", methods=["GET"])
def get_all_entities():
    if not driver:
        return jsonify({"error": "数据库连接失败"}), 500

    try:
        with driver.session() as session:
            movies = [r["name"] for r in session.run("MATCH (m:Movie) RETURN m.name AS name") if r["name"]]
            persons = [r["name"] for r in session.run("MATCH (p:Person) RETURN p.name AS name") if r["name"]]
            companies = [r["name"] for r in session.run("MATCH (c:Company) RETURN c.name AS name") if r["name"]]
            places = [r["name"] for r in session.run("MATCH (p:Place) RETURN p.name AS name") if r["name"]]

        all_names = sorted(set(movies + persons + companies + places))
        return jsonify({
            "success": True,
            "count": len(all_names),
            "names": all_names
        })
    except Exception as e:
        return jsonify({"error": str(e), "names": []}), 500


# ========== 启动 ==========
if __name__ == "__main__":
    print("🚀 启动 Flask 服务...")
    app.run(debug=True, host="127.0.0.1", port=8848)