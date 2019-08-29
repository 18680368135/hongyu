package com.imooc.spark

import org.apache.spark.sql.SparkSession

object hiveMysqlApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("hvieMysqlApp").master("local[2]").getOrCreate()

    //加载hive表数据
    val hiveDF = spark.table("emp")

    //加载mysql表数据
    val mysqlDF = spark.read.format("jdbc").option("url", "jdbc:mysql://g02:3306").option("dbtable", "spark.dept").option("user", "root").option("password", "root").option("driver","com.mysql.jdbc.Driver").load()

    val joinDF = hiveDF.join(mysqlDF,hiveDF.col("deptno")=== mysqlDF.col("deptno"))
    joinDF.show()
    joinDF.select(hiveDF.col("empno"), hiveDF.col("ename"),
      mysqlDF.col("deptno"), mysqlDF.col("dname")).show()
    spark.stop()
  }

}
