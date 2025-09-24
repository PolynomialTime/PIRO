#!/usr/bin/env python3
"""
Process evaluation results CSV files.
For each CSV file in eval_results directory:
1. Keep only the episode_reward column
2. Add a new column 'round' starting from 0
3. Save as new CSV with columns: round, reward

This script ONLY processes CSV files. Use analyze_results.py for statistical analysis.
"""

import os
import pandas as pd
from pathlib import Path

def process_csv_file(csv_path, round_number):
    """Process a single CSV file to extract episode_reward and add round column."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if episode_reward column exists
        if 'episode_reward' not in df.columns:
            print(f"Warning: 'episode_reward' column not found in {csv_path}")
            return False
        
        # Create new DataFrame with only episode_reward
        new_df = pd.DataFrame({
            'round': [round_number] * len(df),
            'reward': df['episode_reward']
        })
        
        # Save the processed file (overwrite original)
        new_df.to_csv(csv_path, index=False)
        print(f"Processed {csv_path} - {len(new_df)} rows, round {round_number}")
        return True
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return False

def main():
    """Process all CSV files in eval_results directory."""
    eval_results_dir = Path('/home/yche767/baseline_new/IQ-Learn/iq_learn/eval_results')
    
    if not eval_results_dir.exists():
        print(f"Error: Directory {eval_results_dir} does not exist")
        return
    
    # Get all environment directories
    env_dirs = [d for d in eval_results_dir.iterdir() if d.is_dir()]
    
    total_processed = 0
    total_files = 0
    
    for env_dir in sorted(env_dirs):
        print(f"\nProcessing environment: {env_dir.name}")
        
        # Get all CSV files in this environment directory
        csv_files = list(env_dir.glob('*.csv'))
        csv_files.sort()  # Sort to ensure consistent ordering
        
        if not csv_files:
            print(f"No CSV files found in {env_dir}")
            continue
        
        # Process each CSV file
        for i, csv_file in enumerate(csv_files):
            total_files += 1
            round_number = i  # Round starts from 0
            
            if process_csv_file(csv_file, round_number):
                total_processed += 1
    
    print(f"\n=== Processing Summary ===")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_files - total_processed}")
    
    # Display a sample of processed files
    print(f"\n=== Sample of processed files ===")
    for env_dir in sorted(env_dirs[:2]):  # Show first 2 environments
        csv_files = list(env_dir.glob('*.csv'))
        if csv_files:
            sample_file = csv_files[0]
            print(f"\n{sample_file}:")
            try:
                df = pd.read_csv(sample_file)
                print(df.head())
            except Exception as e:
                print(f"Error reading sample: {e}")

if __name__ == "__main__":
    main()
