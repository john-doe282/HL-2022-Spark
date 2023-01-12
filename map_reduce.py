import argparse
import json
import os.path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import expr, hour, count
from pyspark.sql.window import Window

from data_generation import read_prepare

RIDES_PATH = 'data/rides.csv'


def top_k_drivers(df: DataFrame, k: int):
    results = (df
               .where(df.driver_rate.isNotNull())
               .orderBy(df.driver_rate, ascending=False)
               .select('driver_id', 'driver_rate')
               .limit(k)
               .rdd
               .map(lambda row: {'driver_id': row[0], 'driver_rate': row[1]})
               )

    return results


def top_k_clients(df: DataFrame, k: int):
    results = (df
               .where(df.client_rate.isNotNull())
               .orderBy(df.client_rate, ascending=False)
               .select('client_id', 'client_rate')
               .limit(k)
               .rdd
               .map(lambda row: {'client_id': row[0], 'client_rate': row[1]})

               )
    return results


def top_k_drivers_by_profit(df: DataFrame, k: int):
    results = (df
               .groupBy(df.driver_id)
               .agg(sum(df.cost).alias('profit'))
               .orderBy('profit', ascending=False)
               .select('driver_id', 'profit')
               .limit(k)
               .rdd
               .map(lambda row: {'driver_id': row[0], 'profit': row[1]})
               )

    return results


def worst_drivers(df: DataFrame, k: int = 100):
    results = (df
               .where(df.driver_rate < 3.5)
               .orderBy(df.driver_rate, ascending=True)
               .select('driver_id', 'driver_rate')
               .limit(k)
               .rdd
               .map(lambda row: {'driver_id': row[0], 'driver_rate': row[1]})

               )

    return results


def top_night_riders(df: DataFrame, k: int):
    df = (df
          .withColumn("hour", hour(df.start_time))
          .withColumn("daytime",
                      expr("case when hour > 0 and hour < 7 "
                           "then 'night' "
                           "else 'day' end"))
          )

    windowSpec_hour = Window.partitionBy("driver_id", "daytime")
    windowSpec = Window.partitionBy("driver_id")

    results = df.withColumn("driver_rides",
                            count(df.client_id).over(windowSpec))

    results = results.withColumn("hour_rides",
                                 count(df.client_id).over(windowSpec_hour))

    results = results.withColumn("pct_rides",
                                 results.hour_rides / results.driver_rides)

    results = (results
               .where(results.daytime == 'night')
               .orderBy('pct_rides', ascending=False)
               .dropDuplicates(['driver_id', 'hour_rides'])
               .select('driver_id', 'hour_rides', 'pct_rides')
               .limit(k)
               .rdd
               .map(lambda row: {'driver_id': row[0],
                                 'night_rides': row[1],
                                 'pct_rides': row[2]})
               )

    return results


def densest_traffic_by_hour(df: DataFrame):
    df = df.withColumn("hour", hour(df.start_time))

    results = (df
               .groupBy(df.hour)
               .agg(count(df.driver_id).alias('count_rides'))
               .orderBy('count_rides', ascending=False)
               .select('hour', 'count_rides')
               .limit(1)
               .rdd
               .map(lambda row: {'hour': f'{row[0]}-{(row[0] + 1) // 24}',
                                 'count_rides': row[1]})
               )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store')
    parser.add_argument('--out', action='store', default='rides.csv')
    parser.add_argument('--num_rows', action='store', type=int, default=1_000_000)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        read_prepare(args.input, args.num_rows).to_csv(args.out, chunksize=100_000)

    spark = SparkSession.builder \
        .master('local[*]') \
        .config('spark.driver.memory', '8g') \
        .getOrCreate()

    df = spark.read.csv(RIDES_PATH, header=True)

    with open('data/top_100_drivers.json', 'w') as f:
        f.write(json.dumps(top_k_drivers(df, 100).collect()))
    with open('data/worst_drivers.json', 'w') as f:
        f.write(json.dumps(worst_drivers(df).collect()))
    with open('data/densest_traffic_by_hour.json', 'w') as f:
        f.write(json.dumps(densest_traffic_by_hour(df).collect()))
    with open('data/top_night_riders.json', 'w') as f:
        f.write(json.dumps(top_night_riders(df, 100).collect()))
