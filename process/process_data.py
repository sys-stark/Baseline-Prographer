import os
import re
import warnings
import torch

def extract_uuid(line):
    pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
    return pattern_uuid.findall(line)

def extract_subject_type(line):
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    return pattern_type.findall(line)

def show(file_path):
    print(f"Processing {file_path}")

def extract_edge_info(line):
    pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    pattern_time = re.compile(r'timestampNanos\":(.*?),')

    edge_type = extract_subject_type(line)[0]
    timestamp = pattern_time.findall(line)[0]
    src_id = pattern_src.findall(line)

    if len(src_id) == 0:
        return None, None, None, None, None

    src_id = src_id[0]
    dst_id1 = pattern_dst1.findall(line)
    dst_id2 = pattern_dst2.findall(line)

    if len(dst_id1) > 0 and dst_id1[0] != 'null':
        dst_id1 = dst_id1[0]
    else:
        dst_id1 = None

    if len(dst_id2) > 0 and dst_id2[0] != 'null':
        dst_id2 = dst_id2[0]
    else:
        dst_id2 = None

    return src_id, edge_type, timestamp, dst_id1, dst_id2


def process_data(file_path):
    id_nodetype_map = {}
    notice_num = 1000000

    with open(file_path, 'r') as f:
        show(file_path)
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(f"processing node {cnt}")

            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                continue

            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                continue

            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                continue

            if 'com.bbn.tc.schema.avro.cdm18.ProvenanceTagNode' in line:
                continue

            uuid = extract_uuid(line)[0]
            subject_type = extract_subject_type(line)

            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid] = 'MemoryObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid] = 'NetFlowObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid] = 'UnnamedPipeObject'
                    continue

            id_nodetype_map[uuid] = subject_type[0]

    return id_nodetype_map

def process_edges_and_count(file_path, id_nodetype_map, output_path):
    notice_num = 1000000
    edge_count = 0

    with open(file_path, 'r') as f, open(output_path, 'a') as fw:
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(f"process_edges {cnt}")

            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(line)

                if src_id is None or src_id not in id_nodetype_map:
                    continue

                src_type = id_nodetype_map[src_id]

                if dst_id1 is not None and dst_id1 in id_nodetype_map:
                    dst_type1 = id_nodetype_map[dst_id1]
                    this_edge1 = f"{src_id}\t{src_type}\t{dst_id1}\t{dst_type1}\t{edge_type}\t{timestamp}\n"
                    fw.write(this_edge1)
                    edge_count += 1

                if dst_id2 is not None and dst_id2 in id_nodetype_map:
                    dst_type2 = id_nodetype_map[dst_id2]
                    this_edge2 = f"{src_id}\t{src_type}\t{dst_id2}\t{dst_type2}\t{edge_type}\t{timestamp}\n"
                    fw.write(this_edge2)
                    edge_count += 1

    return edge_count


def collect_json_paths(base_dir):
    result = {}
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            result[subdir] = {"benign": [], "malicious": []}
            for category in ["benign", "malicious"]:
                category_path = os.path.join(subdir_path, category)
                if os.path.exists(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith(".json") and not file.startswith("._"):
                            full_path = os.path.join(category_path, file)
                            result[subdir][category].append(full_path)
    return result

def run_data_processing():
    # 更换数据集
    # base_path = "../data_files/process"
    # base_path = "../data_files/trace"
    # base_path = "../data_files/cadets"
    base_path = "../data_files/clearscope"

    json_map = collect_json_paths(base_path)
    # 统计良性（benign）和恶意（malicious）的节点数和边数
    statistics = {
        "benign": {"nodes": 0, "edges": 0},
        "malicious": {"nodes": 0, "edges": 0}
    }

    for scene, data in json_map.items():
        for category in ["benign", "malicious"]:
            output_path = os.path.join(base_path, f"{scene}_{category}.txt")
            with open(output_path, 'w') as f:
                for path in data.get(category, []):
                    id_nodetype_map = process_data(path)
                    statistics[category]["nodes"] += len(id_nodetype_map)
                    # 处理边，同时统计边
                    edge_cnt = process_edges_and_count(path, id_nodetype_map, output_path)
                    statistics[category]["edges"] += edge_cnt
    # 统计边/点
    print("\n=========良性/恶意 总节点数与边数统计=========")
    for category in ["benign", "malicious"]:
        print(f"{category}: 节点数 {statistics[category]['nodes']}, 边数 {statistics[category]['edges']}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =================处理数据 边/点=========================
    run_data_processing()