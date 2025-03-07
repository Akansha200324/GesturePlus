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

# [Previous SessionManager and SystemMetricsCollector classes remain the same]

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

# [Rest of the code remains the same]