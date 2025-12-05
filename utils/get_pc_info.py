"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/5 16:56
Description: 
    

"""
import platform
import psutil
import cpuinfo

def get_info():
    info = {}

    # OS 信息
    info["system"] = platform.system()
    info["machine"] = platform.machine()      # 处理器类型
    info["processor"] = cpuinfo.get_cpu_info()['brand_raw']  # CPU 名称 (Intel/Apple M series)

    # CPU 信息
    info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    info["cpu_cores_logical"] = psutil.cpu_count(logical=True)

    # 内存信息
    mem = psutil.virtual_memory()
    info["memory_total_GB"] = round(mem.total / 1024**3, 2)

    return info

if __name__ == "__main__":
    data = get_info()
    for k, v in data.items():
        print(f"{k}: {v}")