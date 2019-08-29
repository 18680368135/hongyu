package com.imooc.log

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


/**
  * TopN 统计spark作业
  */
object TopNStatJob {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("TopNStatJob").config("spark.sql.sources.partitionColumnTypeInference.enabled","false").master("local[2]").getOrCreate()
    val cleanDF = spark.read.format("parquet").load("hdfs://g02:9000/output/clean")
    cleanDF.printSchema()
    cleanDF.show(false)
    videoAccessTopNStat(spark, cleanDF)
    spark.stop()
  }
  def videoAccessTopNStat(spark: SparkSession, cleanDF: DataFrame): Unit ={
//    import spark.implicits._
//    val videoAccessTopNStat = cleanDF.filter($"day" === "20170511" && $"cmsType" === "video").groupBy($"day",$"cmsId").agg(count($"cmsID").as("cmsIdNum")).orderBy($"cmsIdNum".desc)
//    videoAccessTopNStat.show()

    cleanDF.createOrReplaceTempView("cleanStat")
    val cleanSQLDF = spark.sql("select day,cmsId,count(1) as times from cleanStat " +
      "where day='20170511' and cmsType='video'" +
      " group by day, cmsId order by times desc")
    cleanSQLDF.show(false)
  }
}
