# object_type_enum.py
from enum import Enum

class ObjectType(Enum):
    SUBJECT_PROCESS = 0
    MemoryObject = 1
    FILE_OBJECT_BLOCK = 2
    NETFLOW_OBJECT = 3
    PRINCIPAL_REMOTE = 4
    PRINCIPAL_LOCAL = 5
    NetFlowObject = 6
