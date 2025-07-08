from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler}

path_map = {
    "theia": "/home/nsas2020/fuzz/Flash-IDS/data_files/theia",
    "cadets": "/home/nsas2020/fuzz/Flash-IDS/data_files/cadets",
    "clearscope": "/home/nsas2020/fuzz/Flash-IDS/data_files/clearscope",
    "trace": "/home/nsas2020/fuzz/Flash-IDS/data_files/trace",
    "atlas": "D:/baseline/atlas_data",
}

def get_handler(name, train):
    cls = handler_map.get(name.lower())
    base_path = path_map.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    return cls(base_path, train)
