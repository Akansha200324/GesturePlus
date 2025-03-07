import psutil
import time
import subprocess
import argparse
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from typing import Dict, List

class SessionManager:
    def __init__(self, base_dir="monitoring_sessions"):
        self.base_dir = base_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
    def get_data_path(self) -> str:
        return os.path.join(self.session_dir, "metrics.csv")
    
    def get_info_path(self) -> str:
        return os.path.join(self.session_dir, "session_info.json")
    
    def save_session_info(self, info: Dict):
        with open(self.get_info_path(), 'w') as f:
            json.dump(info, f, indent=4)

class SystemMetricsCollector:
    def __init__(self, process):
        self.process = process
        self.has_io_counters = hasattr(process, 'io_counters')
        if self.has_io_counters:
            try:
                self.io_counters_prev = process.io_counters()
                self.timestamp_prev = time.time()
            except (psutil.AccessDenied, AttributeError):
                self.has_io_counters = False

    def collect_metrics(self) -> Dict:
        try:
            metrics = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'cpu_percent': self.process.cpu_percent(),
                'memory_percent': self.process.memory_percent()
            }
            
            # Memory metrics
            memory_info = self.process.memory_info()
            metrics.update({
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
            })
            
            # CPU times
            try:
                cpu_times = self.process.cpu_times()
                metrics.update({
                    'cpu_user': cpu_times.user,
                    'cpu_system': cpu_times.system,
                })
            except (psutil.AccessDenied, AttributeError):
                metrics.update({
                    'cpu_user': 0,
                    'cpu_system': 0,
                })

            # I/O metrics
            if self.has_io_counters:
                try:
                    io_counters = self.process.io_counters()
                    io_delta = time.time() - self.timestamp_prev
                    
                    read_speed = (io_counters.read_bytes - self.io_counters_prev.read_bytes) / io_delta / 1024 / 1024
                    write_speed = (io_counters.write_bytes - self.io_counters_prev.write_bytes) / io_delta / 1024 / 1024
                    
                    self.io_counters_prev = io_counters
                    self.timestamp_prev = time.time()
                    
                    metrics.update({
                        'io_read_mb': read_speed,
                        'io_write_mb': write_speed,
                    })
                except (psutil.AccessDenied, AttributeError):
                    self.has_io_counters = False
            
            if not self.has_io_counters:
                metrics.update({
                    'io_read_mb': 0,
                    'io_write_mb': 0,
                })

            # GPU metrics
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                metrics.update({
                    'gpu_usage': gpus[0].load * 100 if gpus else 0,
                    'gpu_memory_mb': gpus[0].memoryUsed if gpus else 0,
                })
            except:
                metrics.update({
                    'gpu_usage': 0,
                    'gpu_memory_mb': 0,
                })
            
            return metrics
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

# def create_visualizations(df: pd.DataFrame, session_dir: str):
#     fig = make_subplots(
#         rows=3, cols=1,
#         subplot_titles=('CPU & Memory Usage', 'I/O Activity', 'GPU Metrics'),
#         vertical_spacing=0.1,
#         heights=[0.4, 0.3, 0.3]
#     )
    
#     # CPU and Memory plot
#     fig.add_trace(
#         go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %'),
#         row=1, col=1
#     )
#     fig.add_trace(
#         go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %'),
#         row=1, col=1
#     )
    
#     # I/O plot
#     if 'io_read_mb' in df.columns:
#         fig.add_trace(
#             go.Scatter(x=df['timestamp'], y=df['io_read_mb'], name='Read MB/s'),
#             row=2, col=1
#         )
#         fig.add_trace(
#             go.Scatter(x=df['timestamp'], y=df['io_write_mb'], name='Write MB/s'),
#             row=2, col=1
#         )
    
#     # GPU plot
#     if 'gpu_usage' in df.columns:
#         fig.add_trace(
#             go.Scatter(x=df['timestamp'], y=df['gpu_usage'], name='GPU %'),
#             row=3, col=1
#         )
#         fig.add_trace(
#             go.Scatter(x=df['timestamp'], y=df['gpu_memory_mb'], name='GPU Memory MB'),
#             row=3, col=1
#         )
    
#     fig.update_layout(height=1200, title_text="System Resource Usage")
#     fig.write_html(os.path.join(session_dir, 'visualization.html'))

def monitor_process(pid: int, session_manager: SessionManager):
    try:
        process = psutil.Process(pid)
        session_info = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pid': pid,
            'process_name': process.name(),
            'command_line': process.cmdline(),
            'status': 'running'
        }
        
        collector = SystemMetricsCollector(process)
        metrics_list = []
        
        while True:
            metrics = collector.collect_metrics()
            if metrics is None:
                break
                
            metrics_list.append(metrics)
            print(f"CPU: {metrics['cpu_percent']}% | Memory: {metrics['memory_percent']}% | "
                  f"RSS: {metrics['memory_rss_mb']:.1f} MB")
            
            time.sleep(1)
            
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return
    finally:
        if 'session_info' in locals():
            session_info['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_info['status'] = 'completed'
            session_manager.save_session_info(session_info)
            
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                df.to_csv(session_manager.get_data_path(), index=False)
                create_visualizations(df, session_manager.session_dir)


def create_visualizations(df: pd.DataFrame, session_dir: str):
    # Create subplots with correct height specifications
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('CPU & Memory Usage', 'I/O Activity', 'GPU Metrics'),
        vertical_spacing=0.2,
        row_heights=[0.4, 0.3, 0.3]  # Changed from 'heights' to 'row_heights'
    )
    
    # CPU and Memory plot
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %'),
        row=1, col=1
    )
    
    # I/O plot
    if 'io_read_mb' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['io_read_mb'], name='Read MB/s'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['io_write_mb'], name='Write MB/s'),
            row=2, col=1
        )
    
    # GPU plot
    if 'gpu_usage' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['gpu_usage'], name='GPU %'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['gpu_memory_mb'], name='GPU Memory MB'),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="System Resource Usage",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="MB/s", row=2, col=1)
    fig.update_yaxes(title_text="Usage", row=3, col=1)

    # Save the figure
    fig.write_html(os.path.join(session_dir, 'visualization.html'))

def run_and_monitor(command: str, session_manager: SessionManager):
    process = subprocess.Popen(command.split())
    print(f"Started process with PID: {process.pid}")
    monitor_process(process.pid, session_manager)
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor process resource usage')
    parser.add_argument('--command', type=str, help='Command to run and monitor')
    parser.add_argument('--pid', type=int, help='PID of existing process to monitor')
    
    args = parser.parse_args()
    
    session_manager = SessionManager()
    print(f"Starting new monitoring session: {session_manager.session_id}")
    
    if args.command:
        run_and_monitor(args.command, session_manager)
    elif args.pid:
        monitor_process(args.pid, session_manager)
    else:
        print("Please provide either --command or --pid")