package com.imooc.spark

import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}

object hiveContextApp {
  def main(args: Array[String]): Unit = {
    //1创建hiveContext
    var sparkConf = new SparkConf()
    var sc = new SparkContext()
    var hiveContext = new HiveContext(sc)
    //2 进行相应的处理

    hiveContext.table("emp").show

    //3 关闭资源
    sc.stop()

  }
}
