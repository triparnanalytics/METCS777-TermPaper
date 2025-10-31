# METCS777-TermPaper
This repository contains the full implementation and experiments for our METCS777 Big Data Analytics term paper. We evaluate how Apache Spark’s RDDs and DataFrames perform across AWS and Google Cloud using large datasets. The study compares processing time, memory use, and cost to identify the most efficient and scalable setup.

This repository contains code versions to be ran in AWS and Google Cloud. 

File | Description 

METCS777-term-paper-code_df_aws.py: Contains the code to run the sample data in AWS EMR
METCS777-term-paper-code_df_gcp.py: Contains code to run the sample data in Google Cloud
hmda_2016_sample.csv: It's the sample data that we use to test our performance in both platforms
METCS777-term-paper-code_rdd_aws.py: Conttains code to run the sample data in AWS
METCS777-term-paper-code_rdd_gcp_implement.py: Contains code to run sammple data in Google Cloud
test.ipynb: Initial code where when we started. Please ignore. 

For AWS Data Frames: We fullfill the arguments with the proper file directory structure. 

s3://term-paper-fall-2025/METCS777-term-paper-code_df_aws.py
s3://term-paper-fall-2025/hmda_2016_sample.csv s3://term-paper-fall-2025/hmda_test

For Google Cloud:
gs://term-project-fall-2025/METCS777-term-paper-code_df_gcp.py
gs://term-project-fall-2025/hmda_2016_sample.csv gs://term-project-fall-2025/hmda_test

For AWS RDDs:
s3://term-paper-fall-2025/METCS777-term-paper-code_df_aws.py
s3://term-paper-fall-2025/hmda_2016_sample.csv s3://term-paper-fall-2025/hmda_test

For Google Cloud:
gs://term-project-fall-2025/METCS777-term-paper-code_df_gcp.py
gs://term-project-fall-2025/hmda_2016_sample.csv gs://term-project-fall-2025/hmda_test





