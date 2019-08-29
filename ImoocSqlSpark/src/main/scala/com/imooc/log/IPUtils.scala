package com.imooc.log

import com.ggstar.util.ip.IpHelper

object IPUtils {
  def getCity(ip:String)={
    IpHelper.findRegionByIp(ip)
  }
  def main(args: Array[String]): Unit = {
    val ip = "58.30.15.255"
    val region = getCity(ip)
    println(region)
  }
}
