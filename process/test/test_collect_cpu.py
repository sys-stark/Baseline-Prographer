import json
import subprocess
import threading
import time
import psutil


# 定义一个类来收集 CPU 使用率
class CPUMonitor:
    def __init__(self, interval=5, output_file="cpu_usage_log.json"):  #此处更改收集CPU的间隔时间(默认单位为秒)，更改输出json文件的名字
        self.interval = interval  # 采样间隔
        self.output_file = output_file
        self.cpu_usage = []
        self.monitoring = False  # 控制监控的标志

    # 定义收集CPU利用率的函数
    def collect_cpu_usage(self):
        while self.monitoring:
            # 获取当前 CPU 使用率（百分比）
            usage = psutil.cpu_percent(interval=self.interval)
            timestamp = time.time()  # 获取时间戳
            self.cpu_usage.append({"timestamp": timestamp, "cpu_percent": usage})
            print(f"CPU Usage: {usage}% at {timestamp}")

    # 定义将数据保存到文件的函数
    def save_to_file(self):
        # 将 CPU 使用率数据保存到文件
        with open(self.output_file, "w") as f:
            json.dump(self.cpu_usage, f, indent=4)
            print(f"CPU usage data saved to {self.output_file}")

    # 启动 CPU 监控线程的函数
    def start_monitoring(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self.collect_cpu_usage)
        self.thread.start()  # 不设置 daemon

    def stop_monitoring(self):
        self.monitoring = False
        self.thread.join()  # 等待线程安全退出

# 对开启的代码启动CPU监控的函数
def run_xxx_with_cpu_monitor():
    # 创建 CPU 监控实例
    cpu_monitor = CPUMonitor(interval=1, output_file="cpu_usage_log.json")
    cpu_monitor.start_monitoring()  # 启动 CPU 监控

    # 执行 xxx 训练或预测代码
    try:
        # 修改此行代码为你实际的 xxx训练或预测命令
        command = "python test.py"
        working_dir = "../"
        subprocess.run(command, shell=True, check=True, cwd=working_dir)  # 执行命令
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running xxx: {e}")

    # 在 xxx 执行完后，停止 CPU 使用率监控
    cpu_monitor.stop_monitoring()

    # 保存 CPU 使用率日志
    cpu_monitor.save_to_file()

if __name__ == "__main__":
    # 执行 xxx 和 CPU 监控
    run_xxx_with_cpu_monitor()
