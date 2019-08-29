package dataBase;

import java.sql.*;

public class MySQLDemo {

    //JDBC驱动名及数据库URL
    static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://localhost:3306/test";

    //数据库的用户名和密码，需要根据自己的设置
    static final String USER = "root";
    static final String PASS = "root";

    public static void main(String args[]) throws ClassNotFoundException, SQLException{
        Connection conn = null;
        Statement state = null;

        //注册MySQL驱动
        Class.forName("com.mysql.jdbc.Driver");

        //打开连接
        System.out.println("连接数据库.............");
        conn = DriverManager.getConnection(DB_URL, USER, PASS);

        //执行查询
        System.out.println("实例化statement对象............");
        state = conn.createStatement();
        String sql;
        sql = "SELECT id, name, url FROM websites";
        ResultSet rs= state.executeQuery(sql);

        //展开结果数据库
        while (rs.next()){
            //通过字段检索
            int id = rs.getInt("id");
            String name = rs.getString("name");
            String url = rs.getString("url");

            //输出数据
            System.out.print("ID : " + id);
            System.out.print("，站点名称 : " + name);
            System.out.print("，站点 ：" + url);
            System.out.println();

            //用完关闭



        }

        rs.close();
        state.close();
        conn.close();
        System.out.println("goodbye");
    }
}
