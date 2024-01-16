from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, year, to_date

import pandas as pd
import matplotlib.pyplot as plt

import findspark

findspark.init()

# Create a Spark session
spark = SparkSession.builder.appName("NYPD_Shooting").getOrCreate()

# Load the dataset
nypd_data = spark.read.csv("NYPD.csv", header=True, inferSchema=True)

# Data Description
print("Dataset Description:")
nypd_data.printSchema()
print("Number of Rows:", nypd_data.count())

# Data Exercise 1: Incidents Count by Borough
result_1 = nypd_data.groupBy("BORO").agg(count("INCIDENT_KEY").alias("IncidentCount"))
result_1.show()

# Parse 'OCCUR_DATE' to a proper date format
nypd_data = nypd_data.withColumn("OCCUR_DATE", to_date("OCCUR_DATE", "MM/dd/yyyy"))

# Data Exercise 2: Distribution of Incidents over Time (by year)
result_2 = nypd_data.select("OCCUR_DATE", "INCIDENT_KEY").groupBy(year("OCCUR_DATE").alias("Year")).agg(
    count("INCIDENT_KEY").alias("IncidentCount")).orderBy("Year")
result_2_pd = result_2.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(result_2_pd["Year"], result_2_pd["IncidentCount"], color='skyblue')
plt.title('Distribution of Incidents Over Years')
plt.xlabel('Year')
plt.ylabel('Incident Count')
plt.show()

# Data Exercise 3: Distribution of Incidents by Murder Flag
result_3 = nypd_data.groupBy("STATISTICAL_MURDER_FLAG").agg(count("INCIDENT_KEY").alias("IncidentCount"))
result_3_pd = result_3.toPandas()

# Filter out values not in [0, 1]
result_3_pd = result_3_pd[result_3_pd['STATISTICAL_MURDER_FLAG'].isin([0, 1])]

plt.figure(figsize=(6, 4))
plt.bar(result_3_pd["STATISTICAL_MURDER_FLAG"], result_3_pd["IncidentCount"], color='skyblue', width=0.5)
plt.title('Distribution of Incidents by Murder Flag')
plt.xlabel('Murder Flag')
plt.ylabel('Incident Count')
plt.xticks(result_3_pd["STATISTICAL_MURDER_FLAG"])  # Force x-axis to display only 0 and 1
plt.show()


# Data Exercise 4: Incident Count by Jurisdiction Code
result_4 = nypd_data.groupBy("JURISDICTION_CODE").agg(count("INCIDENT_KEY").alias("IncidentCount")).orderBy(
    "IncidentCount", ascending=False)
result_4.show()

# Data Exercise 5: Visualizing Incident Location on Map
result_5 = nypd_data.select("Longitude", "Latitude").filter(col("Longitude").isNotNull() & col("Latitude").isNotNull())
result_5_pd = result_5.toPandas()

plt.figure(figsize=(10, 8))
plt.scatter(x="Longitude", y="Latitude", data=result_5_pd, alpha=0.1)
plt.title('Incident Locations on Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
