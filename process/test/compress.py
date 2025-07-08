# =================训练=========================
import os
from process.datahandlers import get_handler
from process.partition import detect_communities


def save_communities_to_txt(allsize, communities, filename="communities_atlas.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for community_id, nodes in communities.items():
            line = f"Community {community_id}: {', '.join(nodes)}\n"
            f.write(line)
    communities_size_bytes = os.path.getsize(filename)

    dot_size_kb = allsize / 1024
    dot_size_mb = dot_size_kb / 1024

    communities_size_kb = communities_size_bytes / 1024
    communities_size_mb = communities_size_kb / 1024
    print("压缩前:")
    print(f"文件大小为：{allsize} 字节")
    print(f"约为：{dot_size_kb:.2f} KB  {dot_size_mb:.2f} MB")
    print("压缩后:")
    print(f"文件 '{filename}' 大小为：{communities_size_bytes} 字节")
    print(f"约为：{communities_size_kb:.2f} KB  {communities_size_mb:.2f} MB")

if __name__ == "__main__":
    data_handler = get_handler("atlas", False)
    data_handler.load()
    allsize = data_handler.total_loaded_bytes
    features, edges, mapp, relations, G = data_handler.build_graph()
    communities = detect_communities(G)
    communities_size_bytes = save_communities_to_txt(allsize, communities)


