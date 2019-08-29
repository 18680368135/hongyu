package com.imooc.spark

import org.apache.spark.sql.SparkSession


/**
  * DataFrame API基本操作
  *
  */
object DataFrameApp {
  def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().appName("DataFrameApp").master("local[2]").getOrCreate()
        val people = spark.read.format("json").load("D:\\people.json")


        people.printSchema()

        //select name from table
        people.select("name").show

        // select name, age+10 as age2 from table
        people.select(people.col("name"), (people.col("age")+10).as("age2")).show()
        //select * from table where age > 19
        people.filter(people.col("age") > 19).show()

        //根据某一列进行分组，然后进行聚合操作  select age,count(1) from table group by age
        people.groupBy("age").count().show()
        people.show()


        spark.close()

  }
}
