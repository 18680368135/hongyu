package com.imooc.spark

import org.apache.spark.sql.SparkSession

object schemaInferAPP {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("chemaInferAPP").master("local[2]").getOrCreate()
    val path = "hdfs://g02:9000/user/json_schema_infer.json"
    val df = spark.read.format("json").load(path)
    df.printSchema()
    df.show()
    spark.stop()
  }
}
