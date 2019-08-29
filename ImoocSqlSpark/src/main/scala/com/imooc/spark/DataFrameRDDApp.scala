package com.imooc.spark


import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}


/**
  * DataFrame 和 RDD的互操作
  */
object DataFrameRDDApp {
  def main(args: Array[String]): Unit = {
      val spark = SparkSession.builder().appName("DataFrameRDDApp").master("local[2]").getOrCreate()

      program(spark)
      //inferReflection(spark)

      spark.stop()
  }


  def program(spark: SparkSession): Unit = {
    val rdd = spark.sparkContext.textFile("hdfs://g02:9000/user/info.txt")
    val info_RDD = rdd.map(_.split(",")).map(line => Row(line(0).toInt, line(1), line(2).toInt))
    val structType = StructType(Array(StructField("id", IntegerType,true), StructField("name", StringType,true), StructField("age", IntegerType,true)))
    val info_DF = spark.createDataFrame(info_RDD, structType)
    info_DF.printSchema()
    info_DF.show()

    //通过DF的API进行操作或者通过sqlAPI进行操作

  }

  def inferReflection(spark: SparkSession): Unit = {
    val rdd = spark.sparkContext.textFile("hdfs://g02:9000/user/info.txt")
    import spark.implicits._
    val InfoDF = rdd.map(_.split(",")).map(line => Info(line(0).toInt, line(1), line(2).toInt)).toDF("newid", "newname", "newage")
    InfoDF.filter(InfoDF.col("newage") > 30).show()
    InfoDF.createOrReplaceTempView("infos")
    spark.sql("select * from infos where newage > 30").show()
    InfoDF.show()
  }

  case class Info(id:Int, name:String, age:Int)
}
