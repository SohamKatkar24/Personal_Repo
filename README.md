This is a Data Engineering project for **Youtube Data Analysis** developed on Amazon Web Services (AWS) portal:

1. The dataset is taken from Kaggle.
2. Created a Data Lake comprising 3 Amazon S3 buckets: Landing Area, Cleansed/Enriched, and Analytics/Reporting.
3. Processed the raw data from the landing area S3 bucket using AWS Glue, storing the cleansed data in another S3 bucket in parquet format.
4. Ran ETL pipelines on the cleansed data, executed SQL queries using AWS Athena, and built the analytical reporting version.
5. Visualized the data using Amazon QuickSight.
6. Configured monitoring and alerting using AWS CloudWatch.
7. Managed access and permissions using AWS Identity & Access Management (IAM).
