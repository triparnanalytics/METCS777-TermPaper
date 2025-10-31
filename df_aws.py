#!/usr/bin/env python3
"""
Memory-safe HMDA Analysis using Spark DataFrames (AWS-ready)
Fully processes large datasets without driver crashes.
Author: Triparna and Steveen
"""

import os, sys, time, psutil, gc, traceback
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, lit, mean

def get_process_memory_gb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    except Exception:
        return 0.0

def main():
    start_time = time.time()
    initial_memory = get_process_memory_gb()

    INPUT_PATH = os.getenv("INPUT_PATH", "s3a://metcs777-termpaper20007657/hmda_2016_nationwide_all-records_labels.csv")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "s3://metcs777-termpaper20007657/df_aws_analysis/")
    SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"))
    SPARK_DEFAULT_PARALLELISM = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "200"))
    EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    COMPUTE_COST_PER_HOUR = float(os.getenv("AWS_NODE_PRICE_PER_HOUR", "0.2356"))
    EMR_NODES = int(os.getenv("EMR_NODES", "3"))

    try:
        spark = SparkSession.builder \
            .appName("HMDA_Analysis_DataFrame_Safe") \
            .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS) \
            .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM) \
            .config("spark.executor.memory", EXECUTOR_MEMORY) \
            .config("spark.driver.memory", DRIVER_MEMORY) \
            .getOrCreate()
        sc = spark.sparkContext

        # Load Data
        if INPUT_PATH.startswith("s3://"):
            INPUT_PATH = "s3a://" + INPUT_PATH[len("s3://"):]
        df = spark.read.csv(INPUT_PATH, header=True, inferSchema=False)
        record_count = df.count()
        print(f"Loaded {record_count} records.")

        # Sample if too large
        if record_count > 5_000_000:
            samp_frac = min(0.3, 100_000 / record_count)
            df = df.sample(False, samp_frac, seed=42)
            record_count = df.count()
            print(f"Sampled down to {record_count} records for processing.")

        # Create target
        df = df.withColumn("target", when((col("action_taken") == "1") | (col("action_taken_name").rlike("(?i)Loan originated")), 1).otherwise(0))

        # Convert numeric columns
        numeric_columns = [c for c in df.columns if c not in ["action_taken_name","target"]]
        for c in numeric_columns:
            df = df.withColumn(c, when(col(c).isNull() | (col(c) == ""), None).otherwise(col(c).cast("double")))

        # Drop columns with >70% missing
        threshold = 0.7 * record_count
        missing_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        cols_to_drop = [c for c, count_missing in missing_counts.items() if c != "target" and count_missing >= threshold]
        if cols_to_drop:
            df = df.drop(*cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns with >70% missing.")

        # Drop action_taken to avoid leakage
        if "action_taken" in df.columns:
            df = df.drop("action_taken")

        # Impute numeric missing with median (Spark)
        numeric_cols_final = [c for c, t in df.dtypes if t in ("double", "int") and c != "target"]
        medians = df.approxQuantile(numeric_cols_final, [0.5]*len(numeric_cols_final), 0.01)
        median_dict = dict(zip(numeric_cols_final, [m[0] for m in medians]))
        for c in numeric_cols_final:
            df = df.withColumn(c, when(col(c).isNull(), median_dict[c]).otherwise(col(c)))

        # Drop constant columns (Spark)
        nunique = df.agg(*(count(col(c)).alias(c) for c in numeric_cols_final)).collect()[0].asDict()
        const_cols = [c for c in numeric_cols_final if nunique[c] <= 1]
        if const_cols:
            df = df.drop(*const_cols)
            numeric_cols_final = [c for c in numeric_cols_final if c not in const_cols]

        # Collect a bounded sample for sklearn
        sample_size = min(50_000, record_count)
        local_pd = df.sample(False, float(sample_size)/record_count, seed=42).toPandas()
        X = local_pd[numeric_cols_final]
        y = local_pd["target"].astype(int)

        # Train/test split
        stratify_y = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_y)

        # Train RandomForest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]) if len(set(y_test))>1 else float('nan')
        feat_imp = sorted(zip(X.columns.tolist(), rf.feature_importances_), key=lambda x: x[1], reverse=True)
        print(f"Accuracy: {acc:.4f}, AUC: {auc if not np.isnan(auc) else 'N/A'}")

        # Report
        total_time = time.time()-start_time
        hours = total_time/3600
        est_compute_cost = EMR_NODES * COMPUTE_COST_PER_HOUR * hours
        peak_memory_gb = max(initial_memory, get_process_memory_gb())
        storage_cost_per_gb_month = float(os.getenv("S3_STORAGE_COST_PER_GB_MONTH","0.023"))
        est_storage_daily = (peak_memory_gb*storage_cost_per_gb_month)/30
        total_estimated_cost = est_compute_cost + est_storage_daily

        report_lines = [
            "=== HMDA Analysis Report (AWS) ===",
            f"Generated at UTC: {datetime.utcnow().isoformat()}Z",
            f"Records processed: {record_count}",
            "",
            "MODEL PERFORMANCE",
            f"Accuracy: {acc:.4f}",
            f"AUC: {auc if not np.isnan(auc) else 'N/A'}",
            "",
            "TOP FEATURES"
        ]
        for f, imp in feat_imp[:10]:
            report_lines.append(f"{f}: {imp:.6f}")
        report_lines.extend([
            "",
            "PROCESSING METRICS",
            f"Total time (s): {total_time:.2f}",
            f"Peak memory (GB): {peak_memory_gb:.2f}",
            f"Compute cost (USD): {est_compute_cost:.4f}",
            f"Storage daily (USD): {est_storage_daily:.6f}",
            f"Total estimate: {total_estimated_cost:.4f}"
        ])
        report_text = "\n".join(report_lines)
        print(report_text)

        # Save report
        try:
            out_base = OUTPUT_PATH
            if out_base.startswith("s3://"):
                out_base = "s3a://" + out_base[len("s3://"):]
            if not out_base.endswith("/"):
                out_base += "/"
            out_path = out_base + f"hmda_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            spark.createDataFrame([report_text], "string").coalesce(1).write.text(out_path)
            print(f"Report saved to: {out_path}")
        except Exception as e:
            print(f"Could not save to S3: {e}. Writing to local fallback.")
            fallback = "/tmp/hmda_report_fallback.txt"
            with open(fallback, "w") as f: f.write(report_text)
            print(f"Saved fallback report at {fallback}")

    except Exception as exc:
        print("FATAL ERROR:", exc)
        traceback.print_exc()
    finally:
        try: spark.stop()
        except: pass
        final_memory = get_process_memory_gb()
        print(f"Total time: {time.time()-start_time:.2f}s, memory used: {final_memory-initial_memory:.2f}GB")

if __name__=="__main__":
    main()
