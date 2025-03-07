import psutil
import time
import subprocess
import argparse
from datetime import datetime

def monitor_process(pid, output_file):
    """
    Monitor CPU and memory usage of a process and write to a file
    """
    try:
        process = psutil.Process(pid)
        with open(output_file, 'w') as f:
            f.write("Timestamp,CPU %,Memory (MB)\n")
            
            while True:
                try:
                    cpu_percent = process.cpu_percent(interval=1)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    line = f"{timestamp},{cpu_percent:.1f},{memory_mb:.1f}\n"
                    f.write(line)
                    print(line, end='')
                    
                except psutil.NoSuchProcess:
                    print(f"Process {pid} has terminated")
                    break
                    
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found")
        return

def run_and_monitor(command, output_file):
    """
    Run a command and monitor its resource usage
    """
    # Start the process
    process = subprocess.Popen(command.split())
    print(f"Started process with PID: {process.pid}")
    
    # Monitor the process
    monitor_process(process.pid, output_file)
    
    # Wait for process to complete
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor process resource usage')
    parser.add_argument('--command', type=str, help='Command to run and monitor')
    parser.add_argument('--pid', type=int, help='PID of existing process to monitor')
    parser.add_argument('--output', type=str, default='resource_usage.csv',
                      help='Output file for resource usage data')
    
    args = parser.parse_args()
    
    if args.command:
        run_and_monitor(args.command, args.output)
    elif args.pid:
        monitor_process(args.pid, args.output)
    else:
        print("Please provide either --command or --pid")
