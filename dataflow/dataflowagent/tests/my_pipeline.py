import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

INPUT_FILE = "/Users/zyd/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl"
OUTPUT_FILE = "/Users/zyd/DataFlow/dataflow/dataflowagent/tests/my_pipeline_output.json"
AGGREGATED_REPORT = "/Users/zyd/DataFlow/dataflow/dataflowagent/tests/aggregated_report.json"


def json_line_parser(filepath):
    """JSON解析算子：按行读取JSON对象，错误时日志记录并跳过"""
    parsed = []
    with open(filepath, encoding='utf-8') as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                logging.warning(f"第{line_no}行为空，跳过。")
                continue
            try:
                obj = json.loads(line)
                parsed.append(obj)
            except json.JSONDecodeError as e:
                logging.error(f"第{line_no}行解析错误: {e}")
    logging.info(f"成功解析 {len(parsed)} 条JSON数据")
    return parsed


def field_value_infer(data):
    """
    字段值推断算子：推测每个字段的类型、常见值、空值比例等
    reasoning #1
    """
    inferred = {}
    field_types = defaultdict(set)
    field_none_count = Counter()
    field_value_counter = defaultdict(Counter)

    for item in data:
        for k, v in item.items():
            if v is None:
                field_none_count[k] += 1
                field_types[k].add(type(None).__name__)
                continue
            field_types[k].add(type(v).__name__)
            # 为常见小字段统计常见值
            if isinstance(v, (str, int, float, bool)):
                field_value_counter[k][v] += 1

    total = len(data)
    for k in field_types:
        none_percent = field_none_count[k] / total if total else 0
        most_common_values = field_value_counter[k].most_common(5)
        inferred[k] = {
            'types': list(field_types[k]),
            'none_percent': round(none_percent, 2),
            'top_values': [x[0] for x in most_common_values]
        }
    logging.info("字段推断分析完成")
    return inferred


def timeseries_aggregate(data, time_field_candidates=None, freq='minute'):
    """
    时间序列聚合算子：按时间戳字段分桶统计数量
    reasoning #2
    """
    # 寻找时间戳字段
    time_fields = time_field_candidates or []
    if not time_fields:
        # 自动推断可能的timestamp字段
        sample = data[0] if data else {}
        for k, v in sample.items():
            if 'time' in k.lower() or 'timestamp' in k.lower():
                time_fields.append(k)
    if not time_fields:
        logging.warning("未找到时间戳字段，跳过聚合")
        return {}
    time_field = time_fields[0]
    logging.info(f"选用时间字段: {time_field}")

    time_format_list = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%s"
    ]

    def try_parse(ts):
        for fmt in time_format_list:
            try:
                if fmt == "%s":
                    # timestamp数字
                    return datetime.fromtimestamp(float(ts))
                return datetime.strptime(str(ts), fmt)
            except Exception:
                continue
        return None

    # 分桶
    counter = Counter()
    parse_fail = 0
    for i, item in enumerate(data, 1):
        ts_val = item.get(time_field)
        if ts_val is None:
            parse_fail += 1
            continue
        parsed_time = try_parse(ts_val)
        if not parsed_time:
            parse_fail += 1
            continue
        if freq == 'minute':
            bucket = parsed_time.strftime('%Y-%m-%d %H:%M')
        elif freq == 'hour':
            bucket = parsed_time.strftime('%Y-%m-%d %H:00')
        else:
            bucket = parsed_time.strftime('%Y-%m-%d')
        counter[bucket] += 1
    logging.info(f"聚合完成，总{sum(counter.values())}条，解析失败{parse_fail}条")
    return dict(counter)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logging.info(f"已写入{path}")


def main():
    logging.info("流水线开始")
    if not Path(INPUT_FILE).exists():
        logging.error(f"输入文件不存在: {INPUT_FILE}")
        return

    # 1. 读取解析数据
    data = json_line_parser(INPUT_FILE)
    if not data:
        logging.error("没有读到有效数据，流水线结束")
        return

    # 2. 字段值推断
    inferred_fields = field_value_infer(data)

    # 3. 时间序列聚合
    time_series = timeseries_aggregate(data)

    # 4. 输出结果
    result = {
        'field_inference': inferred_fields,
        'time_series_aggregation': time_series,
    }
    save_json(result, OUTPUT_FILE)

    # 附加聚合报告
    save_json({
        "field_summary": inferred_fields,
        "time_series": time_series,
        "total_records": len(data),
    }, AGGREGATED_REPORT)

    logging.info("流水线全部结束")


if __name__ == '__main__':
    main()