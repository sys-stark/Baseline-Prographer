# object_type_enum.py
from enum import Enum

class ObjectType(Enum):
    SUBJECT_PROCESS = 0  # 与进程相关
    MemoryObject = 1  # 与内存块或内存区域相关
    FILE_OBJECT_BLOCK = 2  # 与文件系统的文件或文件块相关
    NETFLOW_OBJECT = 3  # 网络流量相关的对象
    PRINCIPAL_REMOTE = 4  # 远程实体
    PRINCIPAL_LOCAL = 5  # 本地实体
    NetFlowObject = 6  # 与网络流量相关
