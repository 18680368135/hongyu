package com.imooc.spark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

object SQLContextApp {
  def main(args: Array[String]): Unit = {
    var path = args(0)
     //1创建相应的sparkContext
    var sparkconf = new SparkConf()
    //在测试和生产环境中，AapName和Master我们是通过脚本进行指定的
    sparkconf.setAppName("SQLContextApp").setMaster("local[2]")
    var sc = new SparkContext(sparkconf)
    var sqlcontext = new SQLContext(sc)

    //2进行相关的处理:json文件
    var people = sqlcontext.read.format("json").load(path)

    people.printSchema()
    people.show()
    //3关闭资源
    sc.stop()
  }
}
