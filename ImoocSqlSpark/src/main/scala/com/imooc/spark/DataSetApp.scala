package com.imooc.spark

import org.apache.spark.sql.SparkSession

object DataSetApp {
  def main(args: Array[String]): Unit = {
      val spark = SparkSession.builder().appName("DataSetApp").master("local[2]").getOrCreate()
      import spark.implicits._
      val path = "hdfs://g02:9000/user/sales.csv"

      val df = spark.read.option("header", "true").option("inferSchema","true").csv(path)
      df.show()

      val ds = df.as[Sales]

      ds.map(line=> line.itemId).show()

      spark.stop()
  }

  case class Sales(transactionId:Int,customerId:Int,itemId:Int,amountPaid:Double)
}
