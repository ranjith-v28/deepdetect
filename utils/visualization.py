"""
Visualization utilities for deepfake detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_confidence_bar(real_prob, fake_prob):
    """
    Create a confidence bar chart.
    
    Args:
        real_prob: Probability of being real
        fake_prob: Probability of being fake
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    categories = ['REAL', 'FAKE']
    values = [real_prob * 100, fake_prob * 100]
    colors = ['#00CEC9', '#FF6B6B']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
        textfont=dict(size=16, family='Arial Black'),
    ))
    
    fig.update_layout(
        title='Detection Confidence',
        yaxis_title='Confidence (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def plot_frame_grid(frames, max_frames=9):
    """
    Create a grid of video frames.
    
    Args:
        frames: List of frame arrays (numpy arrays)
        max_frames: Maximum number of frames to display
    
    Returns:
        Plotly figure
    """
    # Limit frames
    frames_to_show = frames[:min(max_frames, len(frames))]
    n_frames = len(frames_to_show)
    
    # Calculate grid dimensions
    n_cols = min(3, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Frame {i+1}' for i in range(n_frames)],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )
    
    for idx, frame in enumerate(frames_to_show):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Heatmap(
                z=frame.mean(axis=2) if len(frame.shape) == 3 else frame,
                colorscale='Viridis',
                showscale=False,
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Sample Video Frames',
        height=300 * n_rows,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Hide axes
    for i in fig['layout']:
        if 'axis' in i or 'xaxis' in i or 'yaxis' in i:
            fig['layout'][i]['visible'] = False
    
    return fig


def plot_mel_spectrogram(mfcc, title='Audio Spectrogram'):
    """
    Create a mel-spectrogram visualization.
    
    Args:
        mfcc: MFCC array (n_mfcc, time_steps)
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=mfcc,
        colorscale='Viridis',
        colorbar=dict(title='Amplitude')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Frequency',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def plot_detection_timeline(results):
    """
    Create a timeline of detection results.
    
    Args:
        results: List of detection results
    
    Returns:
        Plotly figure
    """
    if not results:
        return None
    
    times = [r.get('timestamp', i) for i, r in enumerate(results)]
    confidences = [r.get('confidence', 0) * 100 for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#6C5CE7', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Detection Confidence Over Time',
        xaxis_title='Time',
        yaxis_title='Confidence (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig