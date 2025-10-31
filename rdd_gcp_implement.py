#!/usr/bin/env python3
"""
Optimized HMDA Data Analysis using PySpark RDD with Performance Metrics
Author: Triparna and Steveen
"""

import sys
import os
import time
import psutil
import random
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# PySpark imports
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SparkSession

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def get_process_memory_gb():
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def print_data_shape(rdd, step_name, row_count=None):
    """Print the shape (rows, columns) of the RDD after a preprocessing step"""
    try:
        if row_count is None:
            row_count = rdd.count()
        # Get one row to count columns
        sample_row = rdd.first()
        col_count = len(sample_row.keys()) if sample_row else 0
        print(f"Shape after {step_name}: {row_count} rows, {col_count} columns")
        return row_count, col_count
    except Exception as e:
        print(f"Warning: Could not get shape after {step_name}: {e}")
        return 0, 0

def main():
    """Optimized HMDA analysis using PySpark RDD with performance tracking"""

    # -----------------------------
    # 1. Define numeric columns manually
    # -----------------------------
    numeric_columns = [
        "as_of_year","respondent_id","agency_code","loan_type","property_type",
        "loan_purpose","owner_occupancy","loan_amount_000s","preapproval",
        "action_taken","msamd","state_code","county_code","census_tract_number",
        "applicant_ethnicity","co_applicant_ethnicity","applicant_race_1",
        "applicant_race_2","applicant_race_3","applicant_race_4","applicant_race_5",
        "co_applicant_race_1","co_applicant_race_2","co_applicant_race_3",
        "co_applicant_race_4","co_applicant_race_5","applicant_sex","co_applicant_sex",
        "applicant_income_000s","purchaser_type","denial_reason_1","denial_reason_2",
        "denial_reason_3","rate_spread","hoepa_status","lien_status","edit_status",
        "sequence_number","population","minority_population","hud_median_family_income",
        "tract_to_msamd_income","number_of_owner_occupied_units",
        "number_of_1_to_4_family_units","application_date_indicator"
    ]

    # -----------------------------
    # 2. Initialize metrics
    # -----------------------------
    start_time = time.time()
    initial_memory = get_process_memory_gb()
    metrics = {'timing': {}, 'memory': {}, 'cost': {}}

    print(f"Analysis started at: {datetime.now()}")
    print(f"Initial memory usage: {initial_memory:.2f} GB")
    print(f"Using {len(numeric_columns)} numeric columns")

    # -----------------------------
    # 3. Initialize Spark
    # -----------------------------
    spark_start = time.time()
    spark = SparkSession.builder.appName("HMDA_Analysis_Optimized").getOrCreate()
    sc = spark.sparkContext

    try:
        spark.conf.set("spark.sql.shuffle.partitions", "200")
    except Exception as e:
        print(f"Note: Could not set shuffle.partitions config: {e}")

    spark_memory = get_process_memory_gb()
    metrics['timing']['spark_init'] = time.time() - spark_start
    metrics['memory']['spark_init'] = spark_memory - initial_memory
    print(f"Spark initialized in {metrics['timing']['spark_init']:.2f}s, Memory: {spark_memory:.2f} GB")

    # -----------------------------
    # 4. Load Data
    # -----------------------------
    path = "gs://metcs777termpaper/"
    input_file = path + "hmda_2016_nationwide_all-records_labels.csv"
    output_file = path + "hmda_rdd_gcp_output.txt"

    print(f"Input data: {input_file}")
    print(f"Output will be saved to: {output_file}")

    load_start = time.time()
    lines = sc.textFile(input_file)
    header = lines.first().split(',')
    data = lines.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    rdd = data.map(lambda l: dict(zip(header, l.split(','))))

    record_count = rdd.count()
    load_memory = get_process_memory_gb()
    metrics['timing']['data_load'] = time.time() - load_start
    metrics['memory']['data_load'] = load_memory - spark_memory
    print(f"Loaded {record_count} records in {metrics['timing']['data_load']:.2f}s")
    print(f"Memory after load: {load_memory:.2f} GB")
    print_data_shape(rdd, "data load")

    # -----------------------------
    # 5. Sampling for large datasets
    # -----------------------------
    if record_count > 1000000:
        if record_count > 10000000:
            sample_fraction = 0.2
        elif record_count > 5000000:
            sample_fraction = 0.3
        else:
            sample_fraction = 0.5
        print(f"Large dataset detected ({record_count} records)")
        rdd = rdd.sample(False, sample_fraction, seed=42)
        record_count = rdd.count()
        print(f"Working with {record_count} sampled records")
        print_data_shape(rdd, "sampling")

    # -----------------------------
    # 6. Target Variable
    # -----------------------------
    target_start = time.time()
    
    # First, let's check what values are in action_taken field
    print("Checking action_taken values...")
    action_taken_sample = rdd.map(lambda row: row.get("action_taken")).takeSample(False, 1000, seed=42)
    action_taken_counts = {}
    for val in action_taken_sample:
        action_taken_counts[val] = action_taken_counts.get(val, 0) + 1
    print(f"Sample action_taken values: {dict(list(action_taken_counts.items())[:10])}")
    
    # Create target: 1 for approved loans (action_taken = 1), 0 for others
    # action_taken = 1 typically means "Loan originated/approved"
    rdd = rdd.map(lambda row: {**row, "target": 1 if str(row.get("action_taken", "")).strip('"') == "1" else 0})
    
    # Check target distribution
    target_counts = rdd.map(lambda row: row.get("target")).countByValue()
    print(f"Target counts in full RDD: {dict(target_counts)}")
    
    if target_counts.get(1, 0) == 0:
        print("ERROR: No positive 'Loan originated' examples found in the dataset. Check 'action_taken' / 'action_taken_name' fields.")
        # Let's also check action_taken_name values
        action_taken_name_sample = rdd.map(lambda row: row.get("action_taken_name")).takeSample(False, 1000, seed=42)
        action_taken_name_counts = {}
        for val in action_taken_name_sample:
            action_taken_name_counts[val] = action_taken_name_counts.get(val, 0) + 1
        print(f"Sample action_taken_name values: {dict(list(action_taken_name_counts.items())[:10])}")
    
    target_memory = get_process_memory_gb()
    metrics['timing']['target_creation'] = time.time() - target_start
    metrics['memory']['target_creation'] = target_memory - load_memory
    print(f"Target variable created in {metrics['timing']['target_creation']:.2f}s")
    print_data_shape(rdd, "target creation")

    # -----------------------------
    # 7. Clean Numeric Data
    # -----------------------------
    clean_start = time.time()
    def clean_numeric(row):
        new_row = {}
        for k, v in row.items():
            if v is None or str(v).strip() in ("", "NA", "NaN", "null", "nan"):
                new_row[k] = None
            else:
                try:
                    new_row[k] = float(v)
                except (ValueError, TypeError):
                    new_row[k] = v
        return new_row

    rdd = rdd.map(clean_numeric)
    rdd.persist(StorageLevel.MEMORY_AND_DISK)
    _ = rdd.count()
    clean_memory = get_process_memory_gb()
    metrics['timing']['data_cleaning'] = time.time() - clean_start
    metrics['memory']['data_cleaning'] = clean_memory - target_memory
    print(f"Data cleaned and cached in {metrics['timing']['data_cleaning']:.2f}s")
    print_data_shape(rdd, "data cleaning")

    # -----------------------------
    # 8. Predefined Numeric Columns
    # -----------------------------
    numeric_cols = numeric_columns.copy()
    sample_row = rdd.first()
    header_keys = list(sample_row.keys()) if sample_row else []
    print(f"Total columns in data: {len(header_keys)}")
    print(f"Using {len(numeric_cols)} predefined numeric columns")
    if len(numeric_cols) > 0:
        print(f"First 20 numeric columns: {numeric_cols[:20]}")
        if len(numeric_cols) > 20:
            print(f"Last 10 numeric columns: {numeric_cols[-10:]}")


    # -----------------------------
    # 10. Missing Value Column Removal (>70% missing)
    # -----------------------------
    # Get memory before missing value removal (corr_memory if correlation was done, otherwise current memory)
    corr_memory = get_process_memory_gb()  # Set to current memory before missing value removal
    
    missing_start = time.time()
    total_rows = rdd.count()
    threshold_ratio = 0.7
    threshold = total_rows * threshold_ratio
    try:
        current_header_keys = list(rdd.first().keys())
    except:
        current_header_keys = []

    columns_to_drop = []
    missing_sample_size = min(50000, total_rows)
    missing_fraction = missing_sample_size / total_rows if total_rows > 0 else 1.0
    missing_sample_rdd = rdd.sample(False, missing_fraction, seed=42).persist(StorageLevel.MEMORY_ONLY)
    _ = missing_sample_rdd.count()

    def count_missing(row):
        result = {}
        for col in current_header_keys:
            val = row.get(col, None)
            if val is None or val == "" or (isinstance(val, str) and val.strip().lower() in ["na","nan","null"]):
                result[col] = 1
            else:
                result[col] = 0
        return result

    zero_value = {col: 0 for col in current_header_keys}
    def seq_op(acc, row_dict):
        for col in current_header_keys:
            acc[col] = acc.get(col, 0) + row_dict.get(col,0)
        return acc
    def comb_op(acc1, acc2):
        for col in current_header_keys:
            acc1[col] = acc1.get(col,0) + acc2.get(col,0)
        return acc1

    missing_counts = missing_sample_rdd.map(count_missing).aggregate(zero_value, seq_op, comb_op)
    missing_sample_rdd.unpersist(blocking=True)
    gc.collect()
    scale_factor = total_rows / missing_sample_size if missing_sample_size > 0 else 1.0

    for col in current_header_keys:
        if col != "target":
            estimated_missing = missing_counts.get(col,0) * scale_factor
            if estimated_missing > threshold:
                columns_to_drop.append(col)

    if columns_to_drop:
        remaining_after_missing = [c for c in numeric_cols if c not in columns_to_drop]
        if len(remaining_after_missing) == 0:
            print(f"WARNING: Missing value removal would remove ALL numeric columns! Skipping.")
            columns_to_drop = []
        else:
            print(f"Dropping {len(columns_to_drop)} columns with >70% missing values: {columns_to_drop[:10]}...")
            rdd = rdd.map(lambda row: {k:v for k,v in row.items() if k not in columns_to_drop})
            numeric_cols = [c for c in numeric_cols if c not in columns_to_drop]
            print(f"Removed {len(columns_to_drop)} columns with majority missing values")
    else:
        print("No columns exceeded 70% missing threshold.")

    # Explicitly remove 'action_taken'
    header_after_drop = list(rdd.first().keys()) if rdd.count()>0 else []
    if "action_taken" in header_after_drop:
        print("Removing 'action_taken' column (explicit exclusion).")
        rdd = rdd.map(lambda row: {k:v for k,v in row.items() if k != "action_taken"})
        numeric_cols = [c for c in numeric_cols if c != "action_taken"]

    missing_count = rdd.count()
    missing_memory = get_process_memory_gb()
    metrics['timing']['missing_value_removal'] = time.time() - missing_start
    metrics['memory']['missing_value_removal'] = missing_memory - corr_memory
    print(f"Removed missing value columns in {metrics['timing']['missing_value_removal']:.2f}s")
    print_data_shape(rdd,"missing value column removal",missing_count)
    print(f"Numeric columns remaining after missing value removal: {len(numeric_cols)}")

    # -----------------------------
    # 11. Prepare data for model
    # -----------------------------
    collect_start = time.time()
    collect_sample_fraction = 0.01
    local_data = rdd.sample(False, collect_sample_fraction, seed=42).collect()
    collect_memory = get_process_memory_gb()
    metrics['timing']['data_collection'] = time.time() - collect_start
    metrics['memory']['data_collection'] = collect_memory - missing_memory
    print(f"Collected data for sklearn in {metrics['timing']['data_collection']:.2f}s")

    # -----------------------------
    # 12. Data Preparation
    # -----------------------------
    exclude_cols = ['respondent_id','edit_status_name','county_name','msamd_name','state_name','state_abbr','target']
    if len(numeric_cols) == 0:
        numeric_cols = numeric_columns.copy()
    final_numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    if len(final_numeric_cols) == 0:
        final_numeric_cols = numeric_cols
    
    # Helper function to convert values to float
    def to_float_safe(val):
        if val is None:
            return None
        # Handle string values - remove quotes if present
        if isinstance(val, str):
            val = val.strip().strip('"').strip("'")
            if val in ('', 'NA', 'NaN', 'null', 'None'):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        # Handle numeric types
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    # Convert data to DataFrame with proper numeric conversion
    X_data = []
    for row in local_data:
        x_row = {}
        for col in final_numeric_cols:
            val = row.get(col)
            x_row[col] = to_float_safe(val)
        X_data.append(x_row)
    
    X = pd.DataFrame(X_data)
    
    # Debug: Check what columns we have
    print(f"DataFrame shape: {X.shape}")
    print(f"DataFrame columns: {list(X.columns)[:10]}...")
    
    # Convert target to int - check actual target values
    y_data = []
    target_counts = {0: 0, 1: 0}
    for row in local_data:
        # Try to get target - could be int, float, or string
        target_raw = row.get("target", 0)
        if isinstance(target_raw, (int, float)):
            target_val = int(target_raw)
        else:
            # Handle string or other types
            target_val = int(to_float_safe(target_raw) or 0)
        y_data.append(target_val)
        target_counts[target_val] = target_counts.get(target_val, 0) + 1
    
    y = pd.Series(y_data)
    print(f"Target distribution: {target_counts}")
    print(f"Target unique values: {y.unique()}")
    
    # Check if we have any valid features
    if X.shape[1] == 0:
        raise ValueError("No features available! X DataFrame is empty.")
    
    # Check for constant features (all same value) and remove them
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant features: {constant_cols[:10]}...")
        X = X.drop(columns=constant_cols)
    
    if X.shape[1] == 0:
        raise ValueError("All features are constant! No variance to learn from.")
    
    # Fill remaining missing values - handle case where median might fail
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            try:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            except Exception:
                X[col] = X[col].fillna(0)
    
    print(f"Final features for training: {X.shape[1]} columns, {X.shape[0]} rows")

    # -----------------------------
    # 13. Model Training
    # -----------------------------
    # Check if we can stratify (need at least 2 samples per class)
    try:
        if len(y.unique()) >= 2 and min(y.value_counts()) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            print("Warning: Cannot stratify, using random split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Warning: Stratification failed: {e}. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution: {dict(y_train.value_counts())}")
    print(f"Test target distribution: {dict(y_test.value_counts())}")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    # Train the model first
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate AUC using probabilities (not predictions)
    # ROC AUC requires probabilities, not class predictions
    try:
        if len(set(y_test)) < 2:
            print("Warning: Only one class present in y_test. ROC AUC cannot be computed.")
            auc = 0.0
        else:
            y_proba = rf.predict_proba(X_test)[:, 1]  # Get probability of positive class
            auc = roc_auc_score(y_test, y_proba)
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc = 0.0
    
    print(f"Random Forest Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Feature importance - use actual feature names from X
    feature_importances = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 features (out of {len(X.columns)}):")
    for feat, imp in feature_importances[:10]:
        print(f"{feat}: {imp:.6f}")
    
    # Debug: Show if importances sum correctly
    total_importance = sum(rf.feature_importances_)
    print(f"Total feature importance sum: {total_importance:.6f}")

    # -----------------------------
    # 14. Report Generation
    # -----------------------------
    report_lines = []
    report_lines.append("=== HMDA Analysis Report ===")
    report_lines.append(f"Report generated at: {datetime.utcnow().isoformat()}Z")
    report_lines.append("")
    report_lines.append(f"Numeric columns used: {len(final_numeric_cols)}")
    report_lines.append(f"Columns dropped due to missing values: {len(columns_to_drop)}")
    report_lines.append(f"Random Forest Accuracy: {acc:.4f}")
    report_lines.append(f"Random Forest AUC: {auc:.4f}")
    report_lines.append("Top 10 features:")
    for feat, imp in feature_importances[:10]:
        report_lines.append(f"{feat}: {imp:.4f}")
    
    # Processing time and memory metrics
    report_lines.append("")
    report_lines.append("=== Processing Metrics ===")
    try:
        total_time = time.time() - start_time
        report_lines.append(f"Total elapsed time (s): {total_time:.2f}")
    except Exception:
        pass
    
    # Timings per step
    if 'timing' in metrics:
        report_lines.append("-- Step Timings (s) --")
        for k, v in metrics['timing'].items():
            report_lines.append(f"{k}: {v:.2f}")
    
    # Memory deltas per step
    if 'memory' in metrics:
        report_lines.append("-- Memory Deltas (GB) --")
        for k, v in metrics['memory'].items():
            report_lines.append(f"{k}: {v:.2f}")
    
    # Current and initial memory
    try:
        current_mem = get_process_memory_gb()
        report_lines.append(f"Current process memory (GB): {current_mem:.2f}")
        report_lines.append(f"Initial process memory (GB): {initial_memory:.2f}")
    except Exception:
        pass
    
    # Cost estimate (compute + storage). Can be overridden via env vars.
    try:
        # Compute cost per hour (Dataproc worker). Default approximates n1-standard-4 on-demand.
        compute_cost_per_hour = float(os.getenv("DATAPROC_COST_PER_HOUR", "0.19"))
        # GCS Standard storage cost per GB-month (approx), used to estimate daily storage cost
        storage_cost_per_gb_month = float(os.getenv("GCS_STORAGE_COST_PER_GB_MONTH", "0.02"))

        hours = (time.time() - start_time) / 3600.0

        # Estimate peak memory footprint seen by the driver process across steps
        try:
            mem_candidates = [
                initial_memory,
                'spark_memory' in locals() and spark_memory or 0.0,
                'load_memory' in locals() and load_memory or 0.0,
                'target_memory' in locals() and target_memory or 0.0,
                'clean_memory' in locals() and clean_memory or 0.0,
                'missing_memory' in locals() and missing_memory or 0.0,
                'collect_memory' in locals() and collect_memory or 0.0,
                'current_mem' in locals() and current_mem or get_process_memory_gb(),
            ]
            peak_memory_gb = max([m for m in mem_candidates if isinstance(m, (int, float))])
        except Exception:
            peak_memory_gb = 'current_mem' in locals() and current_mem or get_process_memory_gb()

        estimated_compute_cost = hours * compute_cost_per_hour
        estimated_storage_cost_daily = (peak_memory_gb * storage_cost_per_gb_month) / 30.0
        total_estimated_cost = estimated_compute_cost + estimated_storage_cost_daily

        metrics['cost']['compute_cost_usd'] = estimated_compute_cost
        metrics['cost']['storage_cost_usd_per_day'] = estimated_storage_cost_daily
        metrics['cost']['total_cost_usd'] = total_estimated_cost
        metrics['cost']['compute_rate_usd_per_hour'] = compute_cost_per_hour
        metrics['cost']['storage_rate_usd_per_gb_month'] = storage_cost_per_gb_month

        report_lines.append("")
        report_lines.append("=== Cost Estimate ===")
        report_lines.append(f"Compute rate (USD/hour): {compute_cost_per_hour:.4f}")
        report_lines.append(f"Runtime (hours): {hours:.4f}")
        report_lines.append(f"Estimated compute cost (USD): {estimated_compute_cost:.4f}")
        report_lines.append(f"Peak driver memory (GB): {peak_memory_gb:.2f}")
        report_lines.append(f"Storage rate (USD/GB-month): {storage_cost_per_gb_month:.4f}")
        report_lines.append(f"Estimated storage cost per day (USD): {estimated_storage_cost_daily:.4f}")
        report_lines.append(f"Total estimated cost (USD): {total_estimated_cost:.4f}")
    except Exception as e:
        report_lines.append("")
        report_lines.append("=== Cost Estimate ===")
        report_lines.append(f"Cost estimation unavailable: {e}")

    # Save report to GCS using Spark RDD
    # Note: saveAsTextFile creates a directory, so the actual file will be in output_file/part-00000
    try:
        # Create RDD with report content and save to GCS
        report_rdd = sc.parallelize(report_lines, numSlices=1)
        report_rdd.coalesce(1).saveAsTextFile(output_file)
        # The actual file will be at: output_file/part-00000
        print(f"Report saved!")
    except Exception as e:
        print(f"Warning: Could not save report to GCS ({output_file}): {e}")
        print("Report content:")
        print("\n".join(report_lines))

    # -----------------------------
    # 15. Cleanup
    # -----------------------------
    rdd.unpersist()
    gc.collect()
    spark.stop()
    print("Analysis completed.")
    total_time = time.time() - start_time
    print(f"Total elapsed time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
