package com.imooc.spark

import org.apache.spark.sql.SparkSession

object sparkSessionApp {
  def main(args: Array[String]): Unit = {
   // var path = args(0)
    var spark = SparkSession.builder().appName("sparkSessionApp").master("local[2]").getOrCreate()

    var people = spark.read.json("D:\\people.json")
    people.select("name").show()

    people.show()


    spark.stop()
  }
}
