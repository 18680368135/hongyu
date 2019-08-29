package com.imooc.log


import org.apache.spark.sql.{Row}
import org.apache.spark.sql.types.{LongType, StringType, StructField,StructType}

/**
  * access转换（输入--》输出）工具类
  */
object AccessConvertUtil {
    //定义输出的字段
    val struct = StructType(
      Array(
        StructField("url",StringType),
        StructField("cmsType",StringType),
        StructField("cmsId",LongType),
        StructField("traffic",LongType),
        StructField("ip",StringType),
        StructField("city",StringType),
        StructField("time",StringType),
        StructField("day",StringType)

      )
    )

    /**
      * 根据输入的每一行数据转换成输出的样式
      * @param log
      */
    def parseLog(log:String) ={
      try {
        val splits = log.split("\t")
        val url = splits(1)
        val traffic = splits(2).toLong
        val ip = splits(3)
        val domain = "http://www.imooc.com/"
        val cms = url.substring(url.indexOf(domain) +domain.length )
        val cmsSplit = cms.split("/")

        var cmsType = ""
        var cmsId = 0l
        if(cmsSplit.length > 1){
          cmsType = cmsSplit(0)
          cmsId = cmsSplit(1).toLong
        }
        val city = IPUtils.getCity(ip)
        val time = splits(0)
        val day = time.substring(0,10).replaceAll("-","")

        Row(url,cmsType, cmsId, traffic, ip, city, time, day)
      }
      catch{
        case e:Exception => Row("","",0l,0l,"","","","")
      }

    }

}
