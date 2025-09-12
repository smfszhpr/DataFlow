import asyncio, csv, io, math, re, ast, statistics, json, datetime
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from dataflow.agent_v2.base.core import BaseTool

# -----------------------------
# 1) CSV 数据画像
# -----------------------------
class CSVProfileTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "csv_profile"
    @classmethod
    def description(cls) -> str: return "对CSV进行快速画像：缺失率、数值列统计、类别Top等"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            path: Optional[str] = Field(default=None, description="CSV 文件路径")
            max_rows: int = 5000
            delimiter: Optional[str] = None
            encoding: str = "utf-8"
            category_topk: int = 10
            # 兼容LLM常用参数格式
            user_message: Optional[str] = Field(default="", description="用户消息，包含CSV路径或需求描述")
        return P

    async def execute(self, path: str = None, max_rows: int = 5000, delimiter: Optional[str]=None,
                      encoding: str="utf-8", category_topk: int = 10, user_message: Optional[str] = "") -> Dict[str, Any]:
        # 参数兼容处理：如果没有path但有user_message，从user_message提取path
        if not path and user_message:
            # 简单提取：假设user_message包含路径
            import re
            path_match = re.search(r'(/[^\s]+\.csv|[A-Za-z]:[^\s]+\.csv)', user_message)
            if path_match:
                path = path_match.group(0)
            else:
                return {"success": False, "error": "无法从消息中提取CSV文件路径"}
        
        if not path:
            return {"success": False, "error": "缺少CSV文件路径"}
        
        # 同步IO在异步函数里是可以接受的（短时），有需要可用线程池
        def profile():
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.read(4096)
                sniff = csv.Sniffer().sniff(sample) if delimiter is None else None
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter or sniff.delimiter)
                cols = reader.fieldnames or []
                stats = {
                    c: dict(
                        total=0, missing=0,
                        numeric_count=0, non_numeric_count=0,
                        _sum=0.0, _sum2=0.0, _min=None, _max=None,
                        categories=Counter(), examples=[]
                    ) for c in cols
                }
                n = 0
                for row in reader:
                    n += 1
                    for c in cols:
                        v = row.get(c, "")
                        st = stats[c]
                        st["total"] += 1
                        if v is None or v == "":
                            st["missing"] += 1
                            continue
                        if len(st["examples"]) < 3:
                            st["examples"].append(v)
                        # 尝试当作数值
                        try:
                            fv = float(v.replace(",", ""))  # 简单处理千分位
                            st["numeric_count"] += 1
                            st["_sum"] += fv
                            st["_sum2"] += fv*fv
                            st["_min"] = fv if st["_min"] is None else min(st["_min"], fv)
                            st["_max"] = fv if st["_max"] is None else max(st["_max"], fv)
                        except Exception:
                            st["non_numeric_count"] += 1
                            if len(st["categories"]) < 5000:  # 防爆内存
                                st["categories"][v] += 1
                    if n >= max_rows:
                        break

                # 收敛
                profile_cols = {}
                for c, st in stats.items():
                    total = st["total"]
                    valid = total - st["missing"]
                    is_numeric = (st["numeric_count"] >= 0.8 * valid) if valid > 0 else False
                    col_info = {
                        "missing_rate": (st["missing"]/total) if total else 0.0,
                        "examples": st["examples"],
                        "is_numeric": is_numeric
                    }
                    if is_numeric and st["numeric_count"] > 0:
                        mean = st["_sum"]/st["numeric_count"]
                        var = (st["_sum2"]/st["numeric_count"] - mean*mean) if st["numeric_count"]>0 else 0.0
                        col_info.update({
                            "count": st["numeric_count"],
                            "mean": mean, "std": math.sqrt(max(var, 0.0)),
                            "min": st["_min"], "max": st["_max"]
                        })
                    else:
                        topk = st["categories"].most_common(category_topk)
                        col_info.update({"top_categories": topk, "distinct_est": len(st["categories"])})
                    profile_cols[c] = col_info
                return {
                    "success": True,
                    "rows_scanned": n,
                    "columns": list(cols),
                    "profile": profile_cols
                }
        return await asyncio.get_event_loop().run_in_executor(None, profile)

# -----------------------------
# 2) CSV 时间列检测
# -----------------------------
class CSVDetectTimeColumnsTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "csv_detect_time_columns"
    @classmethod
    def description(cls) -> str: return "采样解析，自动检测CSV中的日期时间列"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            path: str
            candidate_columns: Optional[List[str]] = None
            max_rows: int = 3000
            encoding: str = "utf-8"
        return P

    async def execute(self, path: str, candidate_columns: Optional[List[str]]=None,
                      max_rows: int=3000, encoding: str="utf-8") -> Dict[str, Any]:
        formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"
        ]
        def is_dt(s: str) -> bool:
            for fmt in formats:
                try:
                    datetime.datetime.strptime(s.strip(), fmt)
                    return True
                except Exception:
                    pass
            return False

        def detect():
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                cols = candidate_columns or (reader.fieldnames or [])
                counts = {c: {"total":0, "ok":0, "examples": []} for c in cols}
                n=0
                for row in reader:
                    n+=1
                    for c in cols:
                        v = (row.get(c) or "").strip()
                        if v:
                            counts[c]["total"] += 1
                            if is_dt(v):
                                counts[c]["ok"] += 1
                                if len(counts[c]["examples"]) < 3:
                                    counts[c]["examples"].append(v)
                    if n >= max_rows:
                        break
                out = []
                for c, st in counts.items():
                    coverage = (st["ok"]/st["total"]) if st["total"] else 0.0
                    if coverage >= 0.7:
                        out.append({"column": c, "coverage": coverage, "examples": st["examples"]})
                return {"success": True, "detected": out}
        return await asyncio.get_event_loop().run_in_executor(None, detect)

# -----------------------------
# 3) 根据画像生成 Vega-Lite 图表配置（前端可直接渲染）
# -----------------------------
class CSVVegaSpecTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "csv_vega_spec"
    @classmethod
    def description(cls) -> str: return "根据指定参数生成 Vega-Lite 图表配置（柱状图/折线图等）"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            path: Optional[str] = Field(default=None, description="CSV 文件路径")
            chart: str = Field(default="bar", description="图表类型：bar(柱状图)|line(折线图)")
            x: Optional[str] = Field(default=None, description="X轴字段名")
            y: Optional[str] = Field(default=None, description="Y轴字段名")
            title: Optional[str] = Field(default=None, description="图表标题")
            # 兼容LLM常用参数格式
            user_message: Optional[str] = Field(default=None, description="用户消息，包含CSV路径或需求描述")
        return P

    async def execute(self, path: str = None, chart: str="bar", x: Optional[str]=None,
                      y: Optional[str]=None, title: Optional[str]=None, user_message: Optional[str]=None) -> Dict[str, Any]:
        # 参数兼容处理：如果没有path但有user_message，从user_message提取path
        if not path and user_message:
            # 简单提取：假设user_message包含路径
            import re
            path_match = re.search(r'(/[^\s]+\.csv|[A-Za-z]:[^\s]+\.csv)', user_message)
            if path_match:
                path = path_match.group(0)
            else:
                return {"success": False, "error": "无法从消息中提取CSV文件路径"}
        
        if not path:
            return {"success": False, "error": "缺少CSV文件路径"}
        
        # 生成 Vega-Lite 规范
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": path},
            "mark": chart,
            "encoding": {},
        }
        if x: spec["encoding"]["x"] = {"field": x, "type": "nominal"}
        if y: spec["encoding"]["y"] = {"field": y, "type": "quantitative"}
        if title: spec["title"] = title
        
        return {"success": True, "vega_lite_spec": spec}

# -----------------------------
# 4) 代码静态检查（AST）
# -----------------------------
class ASTStaticCheckTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "code_static_check"
    @classmethod
    def description(cls) -> str: return "AST 静态检查：函数、导入、近似复杂度、TODO 扫描"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            code: str
        return P

    async def execute(self, code: str) -> Dict[str, Any]:
        tree = ast.parse(code)
        functions, imports = [], []
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):
                complexity += 1
            if isinstance(node, ast.FunctionDef):
                functions.append({"name": node.name, "args": [a.arg for a in node.args.args]})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None)
                names = [n.name for n in node.names]
                imports.append({"module": mod, "names": names})
        todo_count = len(re.findall(r"#\s*TODO\b", code))
        return {
            "success": True,
            "functions": functions,
            "imports": imports,
            "approx_complexity": complexity,
            "todo_count": todo_count,
            "warnings": [
                *([f"高复杂度：{complexity}"] if complexity >= 20 else []),
                *([f"存在 {todo_count} 个 TODO 注释"] if todo_count else [])
            ]
        }

# -----------------------------
# 5) 生成 pytest 单测骨架
# -----------------------------
class UnitTestStubTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "code_test_stub"
    @classmethod
    def description(cls) -> str: return "从代码中提取函数并生成 pytest 单测骨架"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            code: str
            module_name: str = "module_under_test"
        return P

    async def execute(self, code: str, module_name: str="module_under_test") -> Dict[str, Any]:
        tree = ast.parse(code)
        tests = ["import pytest", f"import {module_name} as MUT", ""]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn = node.name
                tests.append(f"def test_{fn}():")
                call_args = ", ".join("None" for _ in node.args.args)  # 占位
                tests.append(f"    # TODO: fill inputs & assertions")
                tests.append(f"    result = MUT.{fn}({call_args})")
                tests.append(f"    assert result is not None")
                tests.append("")
        test_code = "\n".join(tests) if len(tests) > 3 else "import pytest\n# no functions found"
        return {"success": True, "test_code": test_code}

# -----------------------------
# 6) 本地检索：构建索引 + 查询（简易倒排 + 词频打分）
# -----------------------------
def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", text)]

class LocalIndexBuildTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "local_index_build"
    @classmethod
    def description(cls) -> str: return "从文档列表构建简易倒排索引，返回可传递的索引对象"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            docs: List[str] = Field(description="文档内容列表")
            titles: Optional[List[str]] = None
        return P

    async def execute(self, docs: List[str], titles: Optional[List[str]]=None) -> Dict[str, Any]:
        index = defaultdict(lambda: defaultdict(int))  # term -> doc_id -> tf
        for i, d in enumerate(docs):
            for tok in _tokenize(d):
                index[tok][i] += 1
        # 压成普通 dict 便于跨工具传递
        inv = {t: dict(post) for t, post in index.items()}
        return {"success": True, "index": {"postings": inv, "doc_count": len(docs), "titles": titles or []}}

class LocalIndexQueryTool(BaseTool):
    @classmethod
    def name(cls) -> str: return "local_index_query"
    @classmethod
    def description(cls) -> str: return "在 local_index_build 返回的索引上做简单检索"
    def params(self) -> type[BaseModel]:
        class P(BaseModel):
            index: Dict[str, Any]
            query: str
            top_k: int = 3
        return P

    async def execute(self, index: Dict[str, Any], query: str, top_k: int=3) -> Dict[str, Any]:
        postings = index.get("postings", {})
        N = index.get("doc_count", 1)
        titles = index.get("titles", [])
        scores = defaultdict(float)
        q_tokens = _tokenize(query)
        # 简单 tf * idf（log缩放）
        for qt in q_tokens:
            post = postings.get(qt, {})
            df = max(1, len(post))
            idf = math.log(N / df)
            for doc_id, tf in post.items():
                scores[doc_id] += (1 + math.log(tf)) * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, sc in ranked:
            results.append({"doc_id": int(doc_id), "score": sc, "title": titles[doc_id] if doc_id < len(titles) else None})
        return {"success": True, "results": results}
