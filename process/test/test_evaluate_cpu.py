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

# 提取时间戳和CPU利用率
def extract_time_and_cpu(data):
    times = [entry['seconds'] for entry in data]
    cpu_percentages = [entry['cpu_percent'] for entry in data]
    return times, cpu_percentages

# 绘制图表
def plot_cpu_usage(times, cpu_percentages):
    plt.plot(times, cpu_percentages)
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage over Time')
    plt.savefig("cpu_pig")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 请替换为你的json文件路径
    file_path = 'cpu_usage_log.json'

    # 1. 加载数据
    data = load_json_file(file_path)

    # 2. 转换时间戳为秒
    data_with_seconds = convert_timestamp_to_seconds(data)

    # 3. 提取时间和CPU使用率
    times, cpu_percentages = extract_time_and_cpu(data_with_seconds)

    # 4. 绘制图表
    plot_cpu_usage(times, cpu_percentages)
