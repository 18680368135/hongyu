package com.imooc.spark

import org.apache.spark.sql.SparkSession

object DataFrameCase {
  def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().appName("DataFrameCase").master("local[2]").getOrCreate()
        val rdd = spark.read.textFile("hdfs://g02:9000/user/student.data")
        import spark.implicits._
        val studentDF = rdd.map(_.split("\\|")).map(line => Student(line(0).toInt, line(1), line(2), line(3))).toDF()
        studentDF.show()
        studentDF.show(30)
        studentDF.show(30,false)
        spark.stop()

  }

  case class Student(id:Int, name:String, phone: String, email:String)
}
