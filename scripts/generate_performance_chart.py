#!/usr/bin/env python3
"""Generate performance charts from benchmark results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path('benchmarks/results')


def collect_results() -> pd.DataFrame:
    result_files = list(RESULTS_DIR.glob('performance_comprehensive_*.json'))
    result_files.extend(RESULTS_DIR.glob('comprehensive_performance_*.json'))
    if len(result_files) < 2:
        print('Not enough historical data for charts')
        return pd.DataFrame()
    data = []
    for file in sorted(result_files):
        with open(file, 'r') as f:
            result = json.load(f)
        version = result.get('metadata', {}).get('datason_version', 'unknown')
        timestamp = result['metadata']['timestamp']
        file_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
        for category, tests in result.get('ml_benchmarks', {}).items():
            for test_name, test_data in tests.items():
                if 'datason_standard' in test_data:
                    short_name = (
                        f"{category.replace('_', ' ').title()}: {test_name.replace('_', ' ').title()}"
                    )
                    data.append({
                        'version': version,
                        'timestamp': timestamp,
                        'date': file_date,
                        'test': short_name,
                        'time_ms': test_data['datason_standard']['mean'] * 1000,
                    })
    df = pd.DataFrame(data)
    if df.empty:
        print('No performance data found')
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    daily_data = []
    for (date, version), group in df.groupby(['date', 'version']):
        for test in group['test'].unique():
            test_data = group[group['test'] == test]
            if not test_data.empty:
                daily_data.append({
                    'version': version,
                    'date': date,
                    'test': test,
                    'time_ms': test_data['time_ms'].mean(),
                })
    return pd.DataFrame(daily_data)


def plot_charts(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    unique_dates = df_clean['date'].nunique()
    unique_versions = df_clean['version'].nunique()
    if unique_dates == 1 and unique_versions == 1:
        print('\U0001F9EA Single-day experimental data detected, using time-based experimental chart')
        fig, ax = plt.subplots(figsize=(16, 10))
        top_tests = df['test'].value_counts().head(3).index
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        for i, test in enumerate(top_tests):
            test_data = df[df['test'] == test].sort_values('timestamp')
            ax.plot(
                test_data['timestamp'],
                test_data['time_ms'],
                marker=markers[i], color=colors[i], linewidth=2,
                markersize=8, label=test, alpha=0.8,
            )
        ax.set_title('Datason Experimental Performance (Development Session)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Time (milliseconds)', fontsize=12)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    else:
        print(f"\U0001F4C8 Multi-version data detected ({unique_versions} versions, {unique_dates} dates), using version-based chart")
        fig, ax = plt.subplots(figsize=(14, 8))
        top_tests = df_clean['test'].value_counts().head(3).index
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        for i, test in enumerate(top_tests):
            test_data = df_clean[df_clean['test'] == test].sort_values('date')
            if len(test_data) > 1:
                ax.plot(
                    test_data['date'],
                    test_data['time_ms'],
                    marker=markers[i], color=colors[i], linewidth=2,
                    markersize=8, label=test, alpha=0.8,
                )
                for _, row in test_data.iterrows():
                    if row['version'] not in ['unknown', '0.4.5']:
                        ax.annotate(
                            f"v{row['version']}",
                            (row['date'], row['time_ms']),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=8, alpha=0.7, ha='center'
                        )
        ax.set_title('Datason Performance Trends by Version', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Time (milliseconds)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        fig.autofmt_xdate()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    output_path = Path('benchmarks/performance_chart.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('\u2705 Performance chart generated with adaptive formatting')


def main() -> None:
    df = collect_results()
    if df.empty:
        return
    print(f"Found {len(df)} data points from {len(df['timestamp'].dt.date.unique())} day(s)")
    print(f"Versions: {sorted(df['version'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    df_clean = aggregate_data(df)
    if df_clean.empty:
        print('No aggregated performance data found')
        return
    print(f"After aggregation: {len(df_clean)} data points")
    plot_charts(df, df_clean)


if __name__ == '__main__':
    main()
