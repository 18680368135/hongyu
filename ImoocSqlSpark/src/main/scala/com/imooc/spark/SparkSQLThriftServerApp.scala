package com.imooc.spark

import java.sql.DriverManager

/**
  * 通过JDBC的方式访问
  */
object SparkSQLThriftServerApp {
  def main(args: Array[String]): Unit = {
    Class.forName("org.apache.hive.jdbc.HiveDriver")
    val conn = DriverManager.getConnection("jdbc:hive2://g02:10000","tom","")
    val pstmt = conn.prepareStatement("select empno, ename, sal from emp")
    val rs = pstmt.executeQuery()
    while (rs.next()){
      println("empno:" + rs.getInt("empno") +
      ", ename: " + rs.getString("ename")  +
      ", sal:" + rs.getDouble("sal"))
    }
    rs.close()
    pstmt.close()
    conn.close()
  }
}
