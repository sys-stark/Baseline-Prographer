import json
import matplotlib.pyplot as plt

# 加载JSON文件
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 转换时间戳为秒数
def convert_timestamp_to_seconds(data):
    # 获取第一个时间戳
    start_timestamp = data[0]['timestamp']

    # 计算每个数据点的秒数
    for i, entry in enumerate(data):
        entry['seconds'] = round(entry['timestamp'] - start_timestamp)

    return data

# 提取时间戳和内存使用量（MB）
def extract_time_and_memory(data):
    times = [entry['seconds'] for entry in data]
    memory_usage = [entry['memory_used_MB'] for entry in data]  # 使用内存的MB数
    return times, memory_usage

# 绘制图表
def plot_memory_usage(times, memory_usage):
    plt.plot(times, memory_usage)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage over Time')
    plt.grid(True)
    plt.savefig("memory_pig")  # 保存图像
    plt.show()

if __name__ == "__main__":
    # 请替换为你的json文件路径
    file_path = 'memory_usage_log.json'

    # 1. 加载数据
    data = load_json_file(file_path)

    # 2. 转换时间戳为秒
    data_with_seconds = convert_timestamp_to_seconds(data)

    # 3. 提取时间和内存使用量（MB）
    times, memory_usage = extract_time_and_memory(data_with_seconds)

    # 4. 绘制图表
    plot_memory_usage(times, memory_usage)
