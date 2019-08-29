package com.imooc.spark

import org.apache.spark.sql.SparkSession


/**
  * parquet 文件操作
  */
object ParquetApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ParquetApp").master("local[2]").getOrCreate()
    val path = "file:///home/tom/spark/examples/src/main/resources/users.parquet"
    val userDF = spark.read.format("parquet").load(path)
    userDF.printSchema()
    userDF.show()
    userDF.select(userDF.col("name"), userDF.col("favorite_color"))


    userDF.select("name", "favorite_color").write.format("json").save("hdfs://g02:9000/out/users.json")
    spark.read.load("file:///home/tom/spark/examples/src/main/resources/users.parquet").show

    spark.sqlContext.setConf("spark.sql.shuffle.partitions","10")
    spark.read.format("parquet").option("path", "file:///home/tom/spark/examples/src/main/resources/users.parquet").load().show()


    spark.stop()
  }


}
