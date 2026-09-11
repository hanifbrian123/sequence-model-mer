import os
import time
import subprocess
import psutil

def is_python_running(exclude_pid):
    for p in psutil.process_iter(['name', 'pid']):
        if p.info['name'] and 'python' in p.info['name'].lower() and p.info['pid'] != exclude_pid:
            return True
    return False

if __name__ == "__main__":
    my_pid = os.getpid()
    print("Waiting for other python processes to finish...")
    while is_python_running(my_pid):
        time.sleep(60)
        
    print("No other python processes found! Starting Batch 2...")
    subprocess.run(["bash", "scripts/run_queue.sh", "202_expC", "203_expD"])
    print("Batch 2 finished.")
