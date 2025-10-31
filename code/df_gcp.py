#!/usr/bin/env python3
"""
Optimized HMDA Data Analysis using PySpark DataFrame (crash-safe)
Author: Triparna and Steveen
"""

import os
import time
import psutil
import gc
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan

def get_process_memory_gb():
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def main():
    start_time = time.time()
    initial_memory = get_process_memory_gb()
    metrics = {'timing': {}, 'memory': {}, 'cost': {}}

    print(f"Analysis started at: {datetime.now()}")
    print(f"Initial memory usage: {initial_memory:.2f} GB")

    # -----------------------------
    # 1. Initialize Spark
    # -----------------------------
    spark_start = time.time()
    spark = SparkSession.builder.appName("HMDA_Analysis_DF").getOrCreate()
    sc = spark.sparkContext
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    spark_memory = get_process_memory_gb()
    metrics['timing']['spark_init'] = time.time() - spark_start
    metrics['memory']['spark_init'] = spark_memory - initial_memory
    print(f"Spark initialized in {metrics['timing']['spark_init']:.2f}s, Memory: {spark_memory:.2f} GB")

    # -----------------------------
    # 2. Load Data
    # -----------------------------
    path = "gs://metcs777termpaper/"
    input_file = path + "hmda_2016_nationwide_all-records_labels.csv"
    output_file = path + "hmda_df_gcp_output.txt"

    load_start = time.time()
    df = spark.read.option("header", True).csv(input_file)
    load_memory = get_process_memory_gb()
    metrics['timing']['data_load'] = time.time() - load_start
    metrics['memory']['data_load'] = load_memory - spark_memory
    print(f"Loaded {df.count()} rows, {len(df.columns)} columns in {metrics['timing']['data_load']:.2f}s")
    
    total_rows = df.count()

    # -----------------------------
    # 3. Sampling if large
    # -----------------------------
    sample_fraction = 1.0
    if total_rows > 10_000_000:
        sample_fraction = 0.2
    elif total_rows > 5_000_000:
        sample_fraction = 0.3
    elif total_rows > 1_000_000:
        sample_fraction = 0.5

    if sample_fraction < 1.0:
        df = df.sample(False, sample_fraction, seed=42)
        total_rows = df.count()
        print(f"Sampled dataset: {total_rows} rows (fraction={sample_fraction})")

    # -----------------------------
    # 4. Target Creation
    # -----------------------------
    df = df.withColumn("target", when(col("action_taken") == "1", 1).otherwise(0))
    target_counts = df.groupBy("target").count().toPandas()
    print(f"Target distribution:\n{target_counts}")

    # -----------------------------
    # 5. Clean Numeric Columns
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

    # Convert numeric columns safely
    for col_name in numeric_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, when(col(col_name).rlike("^[0-9.+-eE]+$"), col(col_name).cast("double")).otherwise(None))
    
    # -----------------------------
    # 6. Remove high missing value columns (>70%)
    # -----------------------------
    missing_threshold = 0.7 * total_rows
    missing_counts = df.select([(count(when(col(c).isNull() | isnan(c), c))).alias(c) for c in df.columns]).collect()[0].asDict()
    columns_to_drop = [k for k,v in missing_counts.items() if v > missing_threshold and k != "target"]
    df = df.drop(*columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns with >70% missing values")

    # Remove 'action_taken' explicitly
    if "action_taken" in df.columns:
        df = df.drop("action_taken")

    # -----------------------------
    # 7. Collect small sample to driver safely
    # -----------------------------
    collect_fraction = 0.01 if df.count() > 100_000 else 1.0
    local_df = df.sample(False, collect_fraction, seed=42).toPandas()
    print(f"Collected {len(local_df)} rows locally for scikit-learn")

    # -----------------------------
    # 8. Prepare Features and Target
    # -----------------------------
    exclude_cols = ['respondent_id','edit_status','target']
    feature_cols = [c for c in local_df.columns if c not in exclude_cols]
    
    X = local_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = local_df["target"].astype(int)

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    print(f"Final features: {X.shape[1]} columns, {X.shape[0]} rows")

    # -----------------------------
    # 9. Model Training
    # -----------------------------
    if len(y.unique()) < 2 or min(y.value_counts()) < 2:
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        if len(set(y_test)) < 2:
            auc = 0.0
        else:
            y_proba = rf.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0

    print(f"Random Forest Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # -----------------------------
    # 10. Feature Importances
    # -----------------------------
    feature_importances = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Top 10 features:")
    for feat, imp in feature_importances[:10]:
        print(f"{feat}: {imp:.6f}")

    # -----------------------------
    # 11. Calculate final metrics
    # -----------------------------
    total_time = time.time() - start_time
    final_memory = get_process_memory_gb()
    peak_memory_gb = max(initial_memory, final_memory, load_memory)
    total_memory_used = final_memory - initial_memory
    
    # Cost estimation (GCP Dataproc)
    compute_cost_per_hour = 0.19  # Default n1-standard-4 on-demand
    storage_cost_per_gb_month = 0.02  # Default GCS Standard
    estimated_compute_cost = 0.0
    estimated_storage_cost_daily = 0.0
    total_estimated_cost = 0.0
    
    try:
        compute_cost_per_hour = float(os.getenv("DATAPROC_COST_PER_HOUR", "0.19"))  # n1-standard-4 on-demand
        storage_cost_per_gb_month = float(os.getenv("GCS_STORAGE_COST_PER_GB_MONTH", "0.02"))  # GCS Standard
        
        hours = total_time / 3600.0
        estimated_compute_cost = hours * compute_cost_per_hour
        estimated_storage_cost_daily = (peak_memory_gb * storage_cost_per_gb_month) / 30.0
        total_estimated_cost = estimated_compute_cost + estimated_storage_cost_daily
        
        metrics['cost']['compute_cost_usd'] = estimated_compute_cost
        metrics['cost']['storage_cost_usd_per_day'] = estimated_storage_cost_daily
        metrics['cost']['total_cost_usd'] = total_estimated_cost
    except Exception as e:
        print(f"Warning: Could not calculate cost estimate: {e}")

    # -----------------------------
    # 12. Report
    # -----------------------------
    report_lines = [
        f"=== HMDA Analysis Report ===",
        f"Report generated at: {datetime.utcnow().isoformat()}Z",
        f"",
        f"MODEL PERFORMANCE",
        f"Random Forest Accuracy: {acc:.4f}",
        f"Random Forest AUC: {auc:.4f}",
        f"",
        f"DATA PROCESSING",
        f"Dropped columns due to missing: {len(columns_to_drop)}",
        f"",
        f"Top 10 features:"
    ] + [f"{f}: {i:.4f}" for f,i in feature_importances[:10]] + [
        f"",
        f"PROCESSING METRICS",
        f"Total elapsed time (s): {total_time:.2f}",
        f"Total elapsed time (minutes): {total_time/60:.2f}",
        f"Peak memory usage (GB): {peak_memory_gb:.2f}",
        f"Total memory used (GB): {total_memory_used:.2f}",
        f"",
        f"COST ESTIMATION",
        f"Compute rate (USD/hour): {compute_cost_per_hour:.4f}",
        f"Estimated compute cost (USD): {estimated_compute_cost:.4f}",
        f"Storage rate (USD/GB-month): {storage_cost_per_gb_month:.4f}",
        f"Estimated storage cost per day (USD): {estimated_storage_cost_daily:.6f}",
        f"Total estimated cost (USD): {total_estimated_cost:.4f}"
    ]

    print("\n".join(report_lines))

    # -----------------------------
    # 13. Save report to GCS
    # -----------------------------
    try:
        report_rdd = sc.parallelize(report_lines, 1)
        # saveAsTextFile creates a directory, so the actual file will be at output_file/part-00000
        report_rdd.coalesce(1).saveAsTextFile(output_file)
        print(f"Report saved to: {output_file}/part-00000")
    except Exception as e:
        print(f"Warning: Could not save report to GCS ({output_file}): {e}")
        print("Report content:")
        print("\n".join(report_lines))

    # -----------------------------
    # 14. Cleanup
    # -----------------------------
    gc.collect()
    spark.stop()
    print(f"Total elapsed time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
