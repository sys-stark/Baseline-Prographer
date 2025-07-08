import psutil
import time
import json
import subprocess
import threading

# 定义一个类来收集内存使用量（MB）
class MemoryMonitor:
    def __init__(self, interval=1, output_file="memory_usage_log.json"):     #更改检测的时间与输出文件
        self.interval = interval  # 采样间隔
        self.output_file = output_file
        self.memory_usage = []
        self.monitoring = False  # 是否正在监控
     #收集内存数据函数
    def collect_memory_usage(self):
        while self.monitoring:
            # 获取当前内存使用量（MB）
            memory_used = psutil.virtual_memory().used / (1024 * 1024)  # 转换为MB
            timestamp = time.time()  # 获取时间戳
            self.memory_usage.append({"timestamp": timestamp, "memory_used_MB": memory_used})
            print(f"Memory Usage: {memory_used:.2f} MB at {timestamp}")
            time.sleep(self.interval)  # 休眠一段时间，避免占用过多CPU
    #将数据保存到一个输出文件
    def save_to_file(self):
        # 将内存使用量数据保存到文件
        with open(self.output_file, "w") as f:
            json.dump(self.memory_usage, f, indent=4)
            print(f"Memory usage data saved to {self.output_file}")
  #打开一个线程开始监控
    def start_monitoring(self):
        # 启动内存监控线程
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.collect_memory_usage)
        monitor_thread.daemon = True  # 允许主程序退出时，后台线程自动退出
        monitor_thread.start()
   #终止监控
    def stop_monitoring(self):
        # 停止内存监控
        self.monitoring = False

# 定义一个函数来启动 xxx并进行内存监控
def run_xxx_with_memory_monitor():
    # 创建内存监控实例
    memory_monitor = MemoryMonitor(interval=1, output_file="memory_usage_log.json")
    memory_monitor.start_monitoring()  # 启动内存监控

    # 执行 python xxx.py训练或预测代码
    try:
        # 修改此行代码为你实际的 xxx训练或预测命令
        command = "python test.py"
        working_dir = "../"
        subprocess.run(command, shell=True, check=True, cwd=working_dir)  # 执行命令
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running xxx: {e}")

    # xxx 执行完后，停止内存监控
    memory_monitor.stop_monitoring()

    # 保存内存使用量日志
    memory_monitor.save_to_file()

if __name__ == "__main__":
    # 执行 xxx 和内存监控
    run_xxx_with_memory_monitor()
