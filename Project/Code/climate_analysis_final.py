from pyspark.sql import SparkSession
from pyspark.sql.functions import year, to_date, col, regexp_replace, when, avg, min as _min, max as _max
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

# Initialize Spark
spark = SparkSession.builder \
    .appName("ClimatePrediction") \
    .getOrCreate()

# Load the data
df = spark.read.csv(
    "GlobalLandTemperaturesByMajorCity.csv",
    header=True, inferSchema=True
)

# Preprocessing
# 1. Convert Date to year
df = df.withColumn("Date", to_date(col("dt"), "yyyy-MM-dd"))
df = df.withColumn("year", year(col("Date")).cast(IntegerType()))

# 2. Parse latitude and longitude to numeric
df = df.withColumn(
    "Latitude_num",
    when(col("Latitude").endswith("S"), -1).otherwise(1) *
    regexp_replace(col("Latitude"), "[NS]", "").cast(DoubleType())
)
df = df.withColumn(
    "Longitude_num",
    when(col("Longitude").endswith("W"), -1).otherwise(1) *
    regexp_replace(col("Longitude"), "[EW]", "").cast(DoubleType())
)

# Filter out rows with null temperature
df = df.filter(col("AverageTemperature").isNotNull())

# Determine overall year range and filter to last 20 years
years_range = df.agg(_min("year").alias("min_year"), _max("year").alias("max_year")).collect()[0]
max_year = years_range["max_year"]
min_year = max_year - 19
print(f"Filtering data from {min_year} to {max_year}")
df = df.filter((col("year") >= min_year) & (col("year") <= max_year))

# Compute average uncertainty per city for future predictions
city_uncert = df.groupBy("City") \
    .agg(avg("AverageTemperatureUncertainty").alias("AvgUncertainty"))

# Extract distinct city coordinates with unambiguous names
city_coords = df.select(
    "City",
    col("Latitude_num").alias("City_Latitude"),
    col("Longitude_num").alias("City_Longitude")
).distinct()

# Join data, uncertainty, and coords; rename target
data = df.join(city_uncert, on="City") \
    .join(city_coords, on="City") \
    .withColumnRenamed("AverageTemperature", "label")

# Index City to numeric categories
city_indexer = StringIndexer(inputCol="City", outputCol="City_index", handleInvalid="keep")
indexer_model = city_indexer.fit(data)
data_indexed = indexer_model.transform(data)
num_categories = len(indexer_model.labels)

# Assemble features
assembler = VectorAssembler(
    inputCols=["year", "AvgUncertainty", "City_Latitude", "City_Longitude", "City_index"],
    outputCol="features"
)
data_prepared = assembler.transform(data_indexed).select("features", "label")

# Split into train/test
train, test = data_prepared.randomSplit([0.8, 0.2], seed=42)

# Define Random Forest regressor with adjusted maxBins
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxBins=num_categories + 1
)

# Create pipeline (assembler already applied)
pipeline = Pipeline(stages=[rf])

# --- Scalability Test ---
def benchmark(n_partitions):
    train_rep = train.repartition(n_partitions)
    start = time.time()
    model = pipeline.fit(train_rep)
    train_time = time.time() - start

    infer_start = time.time()
    # force full evaluation
    _ = model.transform(test.repartition(n_partitions)).count()
    infer_time = time.time() - infer_start

    return train_time, infer_time

print("\n--- Scalability Benchmark (Train & Inference times) ---")
for p in [2, 4, 8, 16]:
    t_train, t_infer = benchmark(p)
    print(f"Partitions={p:2d} | Train(s)={t_train:.2f} | Inference(s)={t_infer:.2f}")

# --- Final Model Fit on full filtered data ---
final_model = pipeline.fit(train)

# Evaluation on test
eval_df = final_model.transform(test)
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
print(f"\nTest RMSE: {evaluator_rmse.evaluate(eval_df):.3f}")
print(f"Test R2:   {evaluator_r2.evaluate(eval_df):.3f}\n")

# Forecast next 20 years
future_years = list(range(max_year + 1, max_year + 21))
years_df = spark.createDataFrame([(y,) for y in future_years], ["year"])
future = years_df.crossJoin(city_coords.join(city_uncert, on="City"))
future_indexed = indexer_model.transform(future)
future_features = assembler.transform(future_indexed)
future_preds = final_model.transform(future_features)

# Display predictions
future_preds.select("City", "year", "prediction").orderBy("City", "year").show(1000)

# Stop Spark
spark.stop()