# Markerless-gait-analysis
This repository contains script use in the paper "Deep-Learning-Based Markerless Pose Estimation Systems in Gait Analysis: DeepLabCut Custom Training and the Refinement Function" 

- Preprocessing: This script performs several preprocessing steps starting from raw data in CSV format. It fills missing values using interpolation, filters the data to remove noise, removes outliers based on a threshold, and saves the cleaned and filtered data in both CSV and PKL formats.
- Gait_analysis: This script processes gait data from a preprocessed dataset, extracting positional data for specific body parts, identifying gait events (heel strikes and toe-offs), calculating gait parameters (e.g., step length, step time, stance time, gait speed, cadence), and exporting the results to Excel and pickle files.
