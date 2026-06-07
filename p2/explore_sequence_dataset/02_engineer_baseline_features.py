"""
=============================================================================
Script 2: Baseline Feature Engineering (With Temporal Interpolation)
=============================================================================
Purpose:
    This script processes the raw 63-coordinate MediaPipe data and distills it 
    down into the exact two geometric features that survived the ablation study 
    in the IEEE SMC workshop paper: 
    1) Palm-Tilt Orientation (Arctangent of Index vs. Pinky MCP)
    2) Temporal Stability (5-frame rolling maximum Euclidean wrist drift)
    
    CRITICAL UPDATE: This version includes a Pandas interpolation block. 
    Because MediaPipe frequently drops tracking (outputting 0.0) during fast 
    egocentric motion, this script linearly interpolates the missing spatial 
    coordinates before calculating the geometry. This prevents massive "fake" 
    wrist drift spikes and smooths the temporal sequences.
    
Inputs:
    - 'p1_dataset_combined_raw.csv': The raw 66-column dataset generated 
      by Script 1.
      
Outputs:
    - 'experiment_a_baseline.csv': A refined 5-column dataset containing 
      the sequence metadata (video_name, frame_number, label) and the 
      two engineered geometric features.
=============================================================================
"""

import pandas as pd
import numpy as np
import os

def calculate_baseline_features(df):
    # ==========================================
    # DATA CLEANING: Interpolate MediaPipe Failures
    # ==========================================
    # Identify only the coordinate columns
    cols_to_fix = [c for c in df.columns if 'landmark' in c]
    
    # Replace MediaPipe's 0.0 failures with NaNs so Pandas can interpolate them
    df[cols_to_fix] = df[cols_to_fix].replace(0.0, np.nan)
    
    # Interpolate linearly within each video sequence to smooth out missing frames
    df[cols_to_fix] = df.groupby('video_name')[cols_to_fix].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Fill any remaining NaNs (e.g., if a video had zero detections at the very edges) with 0.0
    df[cols_to_fix] = df[cols_to_fix].fillna(0.0)

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    features = pd.DataFrame()
    
    # Preserve sequence tracking and labels
    if 'video_name' in df.columns: features['video_name'] = df['video_name']
    if 'frame_number' in df.columns: features['frame_number'] = df['frame_number']
    if 'label' in df.columns: features['label'] = df['label']
    
    # FEATURE 1: Palm-Tilt Orientation
    # Math: Arctangent of differentials between Index MCP (5) and Pinky MCP (17)
    dx = df['landmark_17_x'] - df['landmark_5_x']
    dy = df['landmark_17_y'] - df['landmark_5_y']
    features['palm_tilt_degrees'] = np.degrees(np.arctan2(dy, dx))
    
    # FEATURE 2: Temporal Stability (Wrist Drift)
    # Math: Max X/Y pixel displacement of Wrist (0) across a 5-frame rolling window
    wrist_dx = df['landmark_0_x'].diff().fillna(0)
    wrist_dy = df['landmark_0_y'].diff().fillna(0)
    euclidean_drift = np.sqrt(wrist_dx**2 + wrist_dy**2)
    
    # FEATURE 3: Reach (Z-axis Differential)
    # Math: Z-coordinate of Middle Fingertip (12) - Z-coordinate of Wrist (0)
    features['reach_z'] = df['landmark_12_z'] - df['landmark_0_z']

    # Use a temporary dataframe to safely apply the rolling window per video clip
    df_temp = pd.DataFrame({'video_name': df['video_name'], 'drift': euclidean_drift})
    
    rolling_max_drift = df_temp.groupby('video_name')['drift'].transform(
        lambda x: x.rolling(window=5, min_periods=1).max()
    )
    
    # Fix the drift for the very first frame of every new video
    first_frames_idx = df_temp.groupby('video_name').head(1).index
    rolling_max_drift.loc[first_frames_idx] = 0.0
    
    features['temporal_stability'] = rolling_max_drift
    
    return features

if __name__ == "__main__":
    RAW_CSV_PATH = "p1_dataset_combined_raw.csv" 
    OUTPUT_CSV_PATH = "experiment_a_baseline.csv"
    
    if os.path.exists(RAW_CSV_PATH):
        print(f"Loading raw MediaPipe data from {RAW_CSV_PATH}...")
        df_raw = pd.read_csv(RAW_CSV_PATH)
        
        df_baseline = calculate_baseline_features(df_raw)
        df_baseline.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print("\nSUCCESS! Baseline features extracted and interpolated.")
        print(f"Processed dataset shape: {df_baseline.shape}")
        print(f"Saved to: {OUTPUT_CSV_PATH}")
    else:
        print(f"ERROR: Could not find raw data at {RAW_CSV_PATH}. Please run Script 1 first.")