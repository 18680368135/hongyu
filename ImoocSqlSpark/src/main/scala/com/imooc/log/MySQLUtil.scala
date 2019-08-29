package com.imooc.log

import java.sql.DriverManager

/**
  * mysql操作工具累
  */
object MySQLUtil {
  /**
    * 获取数据库连接
    * @param args
    */
  def main(args: Array[String]): Unit = {
    def getConnection(): Unit ={
      Connection con =DriverManager.getConnection("mysql://g02:3306","root","root")
    }
  }
}
