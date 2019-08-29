package com.imooc.log

import org.apache.spark.sql.SparkSession

object SparkStatFormatJob {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("SparkStatFotmatJob").master("local[2]").getOrCreate()

    val access = spark.sparkContext.textFile("hdfs://g02:9000/log/access.10000.log")

    access.map(line =>{
      val sprits = line.split(" ")
      val ip = sprits(0)
      val time = sprits(3) +" "+ sprits(4)
      val url = sprits(11).replace("\"","")
      val traffic = sprits(9)


      DateUtils.parse(time) + "\t" + url + "\t" + traffic + "\t" + ip
    }).saveAsTextFile("hdfs://g02:9000/output/imooc/log1")


    //access.take(10).foreach(println)
    spark.stop()

  }

}
