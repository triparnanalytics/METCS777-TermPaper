#!/usr/bin/env python3
"""
Optimized HMDA Data Analysis using PySpark RDD with Performance Metrics (AWS-ready)
Author: Triparna and Steveen (AWS-adapted)
Notes:
 - Uses s3a:// paths by default
 - Auto-selects S3 credential provider (env keys or IAM role)
 - Conservative sampling and Spark configs to avoid OOMs
 - Defensive try/except and guaranteed spark.stop()
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
import traceback

# PySpark imports
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SparkSession

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---- Helpers ----
def get_process_memory_gb():
    """Get current process memory usage in GB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    except Exception:
        return 0.0

def print_data_shape(rdd, step_name, row_count=None):
    """Print the shape (rows, columns) of the RDD after a preprocessing step"""
    try:
        if row_count is None:
            row_count = rdd.count()
        # Get one row to count columns
        sample_row = None
        try:
            sample_row = rdd.first()
        except Exception:
            sample_row = None
        col_count = len(sample_row.keys()) if sample_row else 0
        print(f"Shape after {step_name}: {row_count} rows, {col_count} columns")
        return row_count, col_count
    except Exception as e:
        print(f"Warning: Could not get shape after {step_name}: {e}")
        return 0, 0

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

# ---- Main ----
def main():
    start_time = time.time()
    initial_memory = get_process_memory_gb()
    metrics = {'timing': {}, 'memory': {}, 'cost': {}}
    sc = None
    spark = None

    # USER TUNABLE ENV VARS
    INPUT_PATH = os.getenv("INPUT_PATH", "s3a://metcs777-termpaper20007657/hmda_2016_nationwide_all-records_labels.csv")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "s3://metcs777-termpaper20007657/rdd_aws_analysis/")  # accepts s3:// or s3a://
    # EMR tuning params (override via env)
    SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"))
    SPARK_DEFAULT_PARALLELISM = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "200"))
    EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    COMPUTE_COST_PER_HOUR = float(os.getenv("AWS_NODE_PRICE_PER_HOUR", "0.2356"))  # user-provided approximation
    EMR_NODES = int(os.getenv("EMR_NODES", "3"))

    try:
        print("="*90)
        print("HMDA RDD Analysis (AWS-ready) — start:", datetime.utcnow().isoformat() + "Z")
        print("="*90)
        print(f"Initial process memory: {initial_memory:.2f} GB")
        print(f"Input path: {INPUT_PATH}")
        print(f"Output base: {OUTPUT_PATH}")
        sys.stdout.flush()

        # --- Spark init with safe defaults ---
        spark_start = time.time()
        spark = SparkSession.builder \
            .appName("HMDA_Analysis_AWS_Optimized") \
            .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS) \
            .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM) \
            .config("spark.executor.memory", EXECUTOR_MEMORY) \
            .config("spark.driver.memory", DRIVER_MEMORY) \
            .config("spark.network.timeout", "600s") \
            .config("spark.executor.heartbeatInterval", "30s") \
            .getOrCreate()
        sc = spark.sparkContext

        # Configure Hadoop S3A credential provider: prefer IAM role unless env keys provided
        hadoop_conf = sc._jsc.hadoopConfiguration()
        try:
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                # Use simple credentials provider if keys provided
                hadoop_conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            else:
                # Prefer instance profile (IAM role attached to EMR/Ec2)
                hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")
            # useful tuning settings (can be adjusted)
            hadoop_conf.set("fs.s3a.multipart.size", str(104857600))  # 100MB
            hadoop_conf.set("fs.s3a.fast.upload", "true")
            # optional: increase retry counts for flaky networks
            hadoop_conf.set("fs.s3a.connection.maximum", "1000")
            hadoop_conf.set("fs.s3a.attempts.maximum", "10")
        except Exception as e:
            print(f"Warning: Could not set hadoop s3a configs: {e}")

        spark_memory = get_process_memory_gb()
        metrics['timing']['spark_init'] = time.time() - spark_start
        metrics['memory']['spark_init'] = spark_memory - initial_memory
        print(f"Spark initialized in {metrics['timing']['spark_init']:.2f}s, process memory delta: {metrics['memory']['spark_init']:.2f} GB")

        # -----------------------------
        # 1. Predefined numeric columns (same as your GCP version)
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
        print(f"Using {len(numeric_columns)} predefined numeric columns")

        # -----------------------------
        # 2. Load data from S3 (s3a)
        # -----------------------------
        load_start = time.time()
        # Accept s3:// or s3a:// in OUTPUT_PATH but FORCE input to use s3a scheme
        if INPUT_PATH.startswith("s3://"):
            INPUT_PATH = "s3a://" + INPUT_PATH[len("s3://"):]
        print(f"Reading from: {INPUT_PATH}")

        lines = sc.textFile(INPUT_PATH)

        # Defensive header parse: use take(20) to find first non-empty line and parse CSV safely
        try:
            head = lines.take(20)
            header_line = None
            for raw in head:
                if raw and raw.strip():
                    header_line = raw
                    break
            if not header_line:
                raise RuntimeError("Empty input file or cannot read header")
            # robust split using csv module for header line
            import csv as _csv
            header = next(_csv.reader([header_line]))
            header = [h.strip() for h in header]
        except Exception as e:
            raise RuntimeError(f"Failed to parse header: {e}")

        # parse full CSV using csv.reader in mapPartitions for robustness
        def parse_partitions(iter_lines):
            import csv as _csv
            reader = _csv.reader(iter_lines)
            for row in reader:
                yield row

        rows_rdd = lines.mapPartitions(parse_partitions)

        # skip header rows equal to the parsed header
        def skip_header_if_equals(iter_rows):
            for r in iter_rows:
                if r == header:
                    continue
                yield r
        data_rows = rows_rdd.mapPartitions(skip_header_if_equals)

        # map lists to dicts (safe padding)
        def row_to_dict(row):
            d = {}
            for i, key in enumerate(header):
                if i < len(row):
                    val = row[i]
                    if val == "":
                        d[key] = None
                    else:
                        d[key] = val
                else:
                    d[key] = None
            return d

        rdd = data_rows.map(row_to_dict)
        rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK)

        record_count = rdd.count()
        load_memory = get_process_memory_gb()
        metrics['timing']['data_load'] = time.time() - load_start
        metrics['memory']['data_load'] = load_memory - spark_memory
        print(f"Loaded {record_count} records in {metrics['timing']['data_load']:.2f}s")
        print_data_shape(rdd, "data load", record_count)

        # -----------------------------
        # 3. Sampling for very large datasets (safe caps)
        # -----------------------------
        if record_count > 10_000_000:
            samp_frac = float(os.getenv("SAMPLE_FRACTION", "0.2"))  # default 20% for huge
        elif record_count > 5_000_000:
            samp_frac = float(os.getenv("SAMPLE_FRACTION", "0.3"))
        elif record_count > 1_000_000:
            samp_frac = float(os.getenv("SAMPLE_FRACTION", "0.5"))
        else:
            samp_frac = 1.0

        if samp_frac < 1.0:
            print(f"Large dataset -> sampling fraction {samp_frac}")
            rdd = rdd.sample(False, samp_frac, seed=42).persist(StorageLevel.MEMORY_AND_DISK)
            sampled_count = rdd.count()
            print(f"Sampled down to {sampled_count} records for processing")
            print_data_shape(rdd, "sampling", sampled_count)

        # -----------------------------
        # 4. Target variable creation
        # -----------------------------
        target_start = time.time()

        def make_target(row):
            # positive if action_taken == 1 OR action_taken_name == "Loan originated"
            at = row.get("action_taken")
            atn = row.get("action_taken_name")
            try:
                if at is not None and str(at).strip().strip('"').strip("'") == "1":
                    return {**row, "target": 1}
            except Exception:
                pass
            try:
                if atn is not None and str(atn).strip().lower() == "loan originated":
                    return {**row, "target": 1}
            except Exception:
                pass
            return {**row, "target": 0}

        rdd = rdd.map(make_target).persist(StorageLevel.MEMORY_AND_DISK)
        tgt_counts = rdd.map(lambda r: r.get("target", 0)).countByValue()
        print("Target distribution (counts):", dict(tgt_counts))
        metrics['timing']['target_creation'] = time.time() - target_start
        metrics['memory']['target_creation'] = get_process_memory_gb() - load_memory

        # -----------------------------
        # 5. Clean numeric data (convert strings to floats where possible)
        # -----------------------------
        clean_start = time.time()
        def clean_numeric(row):
            out = {}
            for k, v in row.items():
                if v is None:
                    out[k] = None
                else:
                    s = v.strip() if isinstance(v, str) else v
                    if s == "" or (isinstance(s, str) and s.lower() in ("na","nan","null","none")):
                        out[k] = None
                    else:
                        try:
                            out[k] = float(s)
                        except Exception:
                            out[k] = v
            return out

        rdd = rdd.map(clean_numeric).persist(StorageLevel.MEMORY_AND_DISK)
        _ = rdd.count()
        metrics['timing']['data_cleaning'] = time.time() - clean_start
        metrics['memory']['data_cleaning'] = get_process_memory_gb() - get_process_memory_gb()
        print(f"Data cleaned and cached in {metrics['timing']['data_cleaning']:.2f}s")
        print_data_shape(rdd, "data cleaning")

        # -----------------------------
        # 6. Missing-value column removal (>70% missing)
        # -----------------------------
        missing_start = time.time()
        total_rows = rdd.count()
        if total_rows == 0:
            raise RuntimeError("No rows found in RDD after cleaning")
        missing_sample_size = min(50000, max(1000, int(total_rows * 0.01)))  # sample at least 1k or 1% up to 50k
        missing_fraction = float(missing_sample_size) / float(total_rows)

        sample_missing_rdd = rdd.sample(False, missing_fraction, seed=42).persist(StorageLevel.MEMORY_ONLY)
        sample_missing_count = sample_missing_rdd.count()

        # Build header keys safely from sample
        sample_row = sample_missing_rdd.take(1)
        if sample_row:
            header_keys = list(sample_row[0].keys())
        else:
            header_keys = header[:]  # fallback to parsed header

        zero_val = {c: 0 for c in header_keys}
        def missing_seq(acc, row):
            for c in header_keys:
                v = row.get(c, None)
                if v is None:
                    acc[c] += 1
                else:
                    if isinstance(v, str) and v.strip().lower() in ("", "na", "nan", "null", "none"):
                        acc[c] += 1
            return acc
        def missing_comb(a, b):
            for c in header_keys:
                a[c] += b.get(c, 0)
            return a

        try:
            missing_counts = sample_missing_rdd.map(lambda r: {c: (1 if (r.get(c) is None or (isinstance(r.get(c), str) and r.get(c).strip().lower() in ('','na','nan','null','none'))) else 0) for c in header_keys}) \
                .aggregate(zero_val, lambda acc, m: {c: acc[c] + m.get(c,0) for c in header_keys}, missing_comb)
        except Exception as e:
            print(f"Warning: missing-count aggregate failed: {e}")
            missing_counts = zero_val.copy()

        sample_missing_rdd.unpersist(blocking=True)
        gc.collect()

        # scale counts to estimate full dataset missing counts
        scale = float(total_rows) / float(sample_missing_count) if sample_missing_count > 0 else 1.0
        cols_to_drop = []
        threshold = total_rows * 0.7
        for c in header_keys:
            est_missing = missing_counts.get(c, 0) * scale
            if c != "target" and est_missing >= threshold:
                cols_to_drop.append(c)

        if cols_to_drop:
            # ensure we don't drop ALL numeric columns
            remaining_numeric = [c for c in numeric_columns if c not in cols_to_drop]
            if len(remaining_numeric) == 0:
                print("Warning: dropping columns would remove all numeric columns - skipping drop")
                cols_to_drop = []
            else:
                print(f"Dropping {len(cols_to_drop)} columns with >70% missing (sample-est): {cols_to_drop[:10]}...")
                rdd = rdd.map(lambda row: {k:v for k,v in row.items() if k not in set(cols_to_drop)}).persist(StorageLevel.MEMORY_AND_DISK)

        # remove action_taken to avoid leakage if present
        try:
            # check quickly and drop
            test_row = rdd.take(1)
            if test_row and "action_taken" in test_row[0]:
                print("Removing 'action_taken' column to avoid leakage.")
                rdd = rdd.map(lambda row: {k:v for k,v in row.items() if k != "action_taken"}).persist(StorageLevel.MEMORY_AND_DISK)
        except Exception:
            pass

        metrics['timing']['missing_value_removal'] = time.time() - missing_start
        metrics['memory']['missing_value_removal'] = get_process_memory_gb() - get_process_memory_gb()
        print_data_shape(rdd, "missing value removal")

        # -----------------------------
        # 7. Prepare local sample for sklearn (bounded)
        # -----------------------------
        collect_start = time.time()
        # Sample 10% of dataset
        sample_take = max(100, int(total_rows * 0.2))
        try:
            # prefer takeSample for randomness; fallback to take
            local_data = rdd.takeSample(False, sample_take, seed=42)
        except Exception:
            local_data = rdd.take(sample_take)
        collect_memory = get_process_memory_gb()
        metrics['timing']['data_collection'] = time.time() - collect_start
        metrics['memory']['data_collection'] = collect_memory - get_process_memory_gb()
        print(f"Collected {len(local_data)} rows for sklearn locally (driver)")
        
        if len(local_data) == 0:
            raise RuntimeError("No data collected for local training")

        # -----------------------------
        # 8. Convert local_data to pandas DataFrame and prepare features/target
        # -----------------------------
        exclude_cols = ['respondent_id','edit_status_name','county_name','msamd_name','state_name','state_abbr','target']
        # attempt to build numeric feature list from sample
        sample_keys = list(local_data[0].keys()) if local_data else header
        candidate_numeric = []
        for key in sample_keys:
            if key in exclude_cols:
                continue
            # check if convertible to float for at least 30% of sample
            parsed = 0
            total = 0
            for r in local_data[:min(200, len(local_data))]:
                total += 1
                val = r.get(key)
                if val is None:
                    continue
                try:
                    float(str(val).strip())
                    parsed += 1
                except Exception:
                    pass
            if total > 0 and parsed / total >= 0.3:
                candidate_numeric.append(key)

        # fallback if none found
        if not candidate_numeric:
            candidate_numeric = [k for k in numeric_columns if k in sample_keys]
        final_numeric_cols = candidate_numeric[:50]  # limit number of features to reasonable number

        # Build pandas DataFrame
        X_list = []
        y_list = []
        for row in local_data:
            row_vals = []
            for c in final_numeric_cols:
                v = row.get(c)
                if v is None:
                    row_vals.append(np.nan)
                else:
                    try:
                        row_vals.append(float(str(v).strip()))
                    except Exception:
                        row_vals.append(np.nan)
            X_list.append(row_vals)
            # target handling: ensure int 0/1
            tgt = row.get("target", 0)
            try:
                if isinstance(tgt, (int, float)):
                    y_list.append(int(tgt))
                else:
                    y_list.append(int(safe_float(tgt) or 0))
            except Exception:
                y_list.append(0)

        X = pd.DataFrame(X_list, columns=final_numeric_cols)
        y = pd.Series(y_list)

        print(f"Local DataFrame shape: {X.shape}")
        print(f"Local target distribution (sample): {y.value_counts().to_dict()}")

        # Clean X: drop constant columns
        const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const_cols:
            print(f"Dropping {len(const_cols)} constant cols from local features: {const_cols[:10]}")
            X = X.drop(columns=const_cols)

        # Fill missing values with median
        for c in X.columns:
            if X[c].isna().all():
                X[c] = 0.0
            else:
                try:
                    med = X[c].median()
                    if pd.isna(med):
                        X[c] = X[c].fillna(0.0)
                    else:
                        X[c] = X[c].fillna(med)
                except Exception:
                    X[c] = X[c].fillna(0.0)

        if X.shape[1] == 0:
            raise RuntimeError("No valid features available for training (local sample)")

        # -----------------------------
        # 9. Train/test split and model training
        # -----------------------------
        try:
            if len(y.unique()) >= 2 and (y.value_counts().min() >= 2):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Warning: train_test_split stratify failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = float('nan')
        try:
            if len(set(y_test)) >= 2:
                y_proba = rf.predict_proba(X_test)[:,1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                print("Warning: only one class in test set; AUC not computed")
        except Exception as e:
            print(f"Warning: could not compute AUC: {e}")

        print(f"Trained RandomForest — Accuracy: {acc:.4f}, AUC: {auc if not np.isnan(auc) else 'N/A'}")

        # Feature importances
        try:
            feat_imp = sorted(zip(X.columns.tolist(), rf.feature_importances_), key=lambda x: x[1], reverse=True)
        except Exception:
            feat_imp = []

        # -----------------------------
        # 10. Report generation (includes runtime + cost)
        # -----------------------------
        total_time = time.time() - start_time
        hours = total_time / 3600.0
        # estimate compute cost as nodes * price_per_hour * hours
        est_compute_cost = EMR_NODES * COMPUTE_COST_PER_HOUR * hours
        # storage estimate: use driver peak memory as storage proxy (conservative)
        try:
            peak_memory_gb = max(initial_memory, get_process_memory_gb())
        except Exception:
            peak_memory_gb = initial_memory or 0.0
        storage_cost_per_gb_month = float(os.getenv("S3_STORAGE_COST_PER_GB_MONTH", "0.023"))
        est_storage_daily = (peak_memory_gb * storage_cost_per_gb_month) / 30.0
        total_estimated_cost = est_compute_cost + est_storage_daily

        report_lines = []
        report_lines.append("=== HMDA Analysis Report (AWS) ===")
        report_lines.append(f"Generated at (UTC): {datetime.utcnow().isoformat()}Z")
        report_lines.append("")
        report_lines.append(f"Input: {INPUT_PATH}")
        report_lines.append(f"Records processed: {record_count}")
        report_lines.append("")
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append(f"AUC: {auc if not np.isnan(auc) else 'N/A'}")
        report_lines.append("")
        report_lines.append("TOP FEATURES")
        for f, imp in feat_imp[:10]:
            report_lines.append(f"{f}: {imp:.6f}")
        report_lines.append("")
        report_lines.append("PROCESSING METRICS")
        report_lines.append(f"Total elapsed time (s): {total_time:.2f}")
        report_lines.append(f"Total memory usage (GB): {peak_memory_gb:.2f}")
        report_lines.append(f"Estimated compute cost (USD): {est_compute_cost:.4f} (nodes={EMR_NODES}, rate={COMPUTE_COST_PER_HOUR}/hr)")
        report_lines.append(f"Estimated storage (daily, USD): {est_storage_daily:.6f}")
        report_lines.append(f"Estimated total (compute + daily storage): {total_estimated_cost:.4f}")

        report_text = "\n".join(report_lines)
        print(report_text)

        # -----------------------------
        # 11. Save report to S3 (s3a://); fallback to local /tmp
        # -----------------------------
        try:
            out_base = OUTPUT_PATH
            if out_base.startswith("s3://"):
                # convert to s3a scheme for Spark's saveAsTextFile
                out_base = "s3a://" + out_base[len("s3://"):]
            if not out_base.endswith("/"):
                out_base = out_base + "/"
            out_path = out_base + f"hmda_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            # Create single-part RDD and save
            sc.parallelize([report_text], 1).coalesce(1).saveAsTextFile(out_path)
            print(f"Report saved to: {out_path}")
        except Exception as e:
            print(f"Warning: could not save report to S3: {e}", file=sys.stderr)
            # fallback to local write
            try:
                fallback = "/tmp/hmda_report_fallback.txt"
                with open(fallback, "w") as fh:
                    fh.write(report_text)
                print(f"Report written to fallback local file: {fallback}")
            except Exception as e2:
                print(f"Failed to write fallback report: {e2}", file=sys.stderr)

        # -----------------------------
        # 12. Cleanup
        # -----------------------------
        try:
            rdd.unpersist()
        except Exception:
            pass
        gc.collect()

    except Exception as exc:
        print("FATAL ERROR during processing:", exc, file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            if sc:
                sc.stop()
        except Exception:
            pass
        print("Spark stopped (if it was running).")
        end_time = time.time()
        final_memory = get_process_memory_gb()
        total_memory_used = final_memory - initial_memory
        print(f"Total time elapsed: {end_time - start_time:.2f}s")
        print(f"Total memory used: {total_memory_used:.2f} GB (from {initial_memory:.2f} GB to {final_memory:.2f} GB)")

if __name__ == "__main__":
    main()
