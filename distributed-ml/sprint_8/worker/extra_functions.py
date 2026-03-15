import psutil
import time
from kubernetes import client, config

def get_system_metrics():
    memory_mb, memory_percent = _get_memory_metrics()
    cpu_percent = _get_cpu_percent()
    return {
        'cpu_usage': cpu_percent,
        'memory_mb': memory_mb,
        'memory_usage': memory_percent
    }

def _get_memory_metrics():
    try:
        # cgroups v1
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
            used_bytes = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            limit_bytes = int(f.read().strip())
    except FileNotFoundError:
        # cgroups v2
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            used_bytes = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            content = f.read().strip()
            limit_bytes = int(content) if content != "max" else None

    memory_mb = used_bytes / (1024 * 1024)
    memory_percent = (used_bytes / limit_bytes * 100) if limit_bytes else 0.0
    return round(memory_mb, 2), round(memory_percent, 2)

def _get_cpu_percent():
    try:
        # cgroups v1
        cpu_file = "/sys/fs/cgroup/cpu/cpuacct.usage"
        with open(cpu_file, "r") as f:
            t1 = int(f.read().strip())
        time.sleep(0.1)
        with open(cpu_file, "r") as f:
            t2 = int(f.read().strip())
        # nanosegundos usados en 100ms -> porcentaje
        percent = ((t2 - t1) / 1e8) * 100
    except FileNotFoundError:
        # cgroups v2
        def read_cpu_usec():
            with open("/sys/fs/cgroup/cpu.stat", "r") as f:
                for line in f:
                    if line.startswith("usage_usec"):
                        return int(line.split()[1])
        t1 = read_cpu_usec()
        time.sleep(0.1)
        t2 = read_cpu_usec()
        percent = ((t2 - t1) / 1e5) * 100

    return round(min(percent, 100.0), 2)