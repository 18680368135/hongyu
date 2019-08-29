package com.imooc.log

import org.apache.spark.sql.{SaveMode, SparkSession}

object SparkStatCleanJob {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("SparkStatCleanJob").master("local[2]").getOrCreate()
    val access = spark.sparkContext.textFile("hdfs://g02:9000/imooc/access.log")

    //RDD ==> DataFrame
    val accessDF = spark.createDataFrame(access.map(line => AccessConvertUtil.parseLog(line)), AccessConvertUtil.struct)
    //access.take(10).foreach(println)

//    accessDF.printSchema()
//    accessDF.show()
    accessDF.coalesce(1).write.format("parquet").partitionBy("day").mode(SaveMode.Overwrite).save("hdfs://g02:9000/output/clean")

    spark.stop()
  }
}
