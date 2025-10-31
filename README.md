# METCS777-TermPaper
This repository contains the full implementation and experiments for our METCS777 Big Data Analytics term paper. We evaluate how Apache Spark’s RDDs and DataFrames perform across AWS and Google Cloud using large datasets. The study compares processing time, memory use, and cost to identify the most efficient and scalable setup.

## Project Overview:

**Project Title**: Big Data Performance Evaluation of RDD vs Dataframe on Cloud Platforms  
**By**: Steveen Vargas and Triparna Kundu 
**Project Purpose:**: Compare Spark RDDs vs. DataFrames on AWS and Google Cloud.
**Motivation**: Performance, memory, and cost vary across platforms and data types.
**Method**: Test large datasets; measure time, memory, cost.
**Outcome:** Find the most efficient, scalable, and cost-effective configuration.

## Dataset Information: Home Mortgage Disclosure Act (HMDA) Dataset
- Published annually by the Consumer Financial Protection Bureau (CFPB). Contains detailed information on mortgage applications and outcomes.
Includes data on:
- Consumer details and applicant demographics
- Loan amount and property characteristics
- Lender information and loan decision (approved, denied, or incomplete)
- Used to analyze lending patterns and promote fair housing practices.

## Dataset Sample
**hmda_2016_sample.csv** - This is the sample dataset that was initially used in our local system to test the model.
**hmda_2016_nationwide_all-records_labels.csv** - This is the large dataset that is used to compute and compare our model performance on cloud platforms.

## Files Description
This repository contains code separate code versions that run on AWS and Google Cloud. 

**Dataframes:** 
- df_aws.py: Contains the code to run the sample data in AWS EMR
- df_gcp.py: Contains code to run the sample data in Google Cloud

**RDDS**
- rdd_aws.py: Conttains code to run the sample data in AWS
- rdd_gcp_implement.py: Contains code to run sammple data in Google Cloud

## Instruction Examples: 
**For AWS Data Frames:** We fullfill the arguments with the proper file directory structure. 
- s3://term-paper-fall-2025/METCS777-term-paper-code_df_aws.py
- s3://term-paper-fall-2025/hmda_2016_sample.csv s3://term-paper-fall-2025/hmda_test

**For Google Cloud:**
- gs://term-project-fall-2025/METCS777-term-paper-code_df_gcp.py
- gs://term-project-fall-2025/hmda_2016_sample.csv gs://term-project-fall-2025/hmda_test

**For AWS RDDs:**
- s3://term-paper-fall-2025/METCS777-term-paper-code_df_aws.py
- s3://term-paper-fall-2025/hmda_2016_sample.csv s3://term-paper-fall-2025/hmda_test

**For Google Cloud:**
- gs://term-project-fall-2025/METCS777-term-paper-code_df_gcp.py
- gs://term-project-fall-2025/hmda_2016_sample.csv gs://term-project-fall-2025/hmda_test

#  HMDA Big Data Analysis Using PySpark (AWS & GCP)

This project performs large-scale analysis on the **Home Mortgage Disclosure Act (HMDA)** dataset using **PySpark RDD** and **DataFrame abstractions**, deployed across **AWS EMR** and **Google Cloud Dataproc** environments.  
The objective is to compare **performance, cost, and efficiency** between data abstractions and cloud platforms.

---

## Environment Setup

Follow the steps below to configure and run the project on cloud platforms.

### AWS EMR Setup

1. **Add Bootstrap Script**
   - Include the `install_lib.sh` file under the **Bootstrap Actions** section during EMR cluster creation.  
   - This script installs all required Python libraries (NumPy, Pandas, PySpark, psutil, etc.).

2. **Set Up Logging**
   - Create a `logs/` folder in your **S3 bucket**.  
   - Add this path under **Cluster Logs** to capture runtime and step logs for monitoring.

3. **Upload and Run Scripts**
   - Upload the following files to your S3 bucket:  
     - `df_aws.py`  
     - `rdd_aws.py`  
   - Add them as **Steps** during cluster creation or submit them after cluster launch via the EMR console.

4. **Accessing Outputs**
   - After processing completes, output files will be generated in your S3 bucket:  
     - `hmda-rdd-aws-output.txt`  
     - `DF_aws.txt`  
   - These contain model metrics, system performance, and resource usage reports.

---

### Google Cloud Dataproc Setup

1. **Upload Files**
   - Upload all `.py` scripts and datasets to your **Google Cloud Storage** bucket:  
     - `rdd_gcp_implement.py`  
     - `df_gcp.py`

2. **Create Cluster**
   - Create a **Dataproc cluster** with PySpark pre-installed, or install dependencies manually.

3. **Submit Job**
   - Provide the dataset path and desired output location as arguments during job submission.  
   - Once completed, retrieve the output files from the bucket.

4. **Retrieve Outputs**
   - Output files generated:  
     - `hmda_rdd_gcp_output.txt`  
     - `df_gcp_output.txt`  
   - These files contain the same performance and accuracy metrics as the AWS results.

---

## How to Run the Code

### On AWS:
```bash
# Execute on AWS EMR
python3 df_aws.py
python3 rdd_aws.py




