package serializable;

import java.io.*;
import java.io.ObjectInputStream.GetField;
import java.io.ObjectOutputStream.PutField;

public class Test implements Serializable {
    private static final long serialVersionUID = 1L;
    private String password = "pass";
    public String getPassword(){
        return password;
    }

    public void setPassword(String password){
        this.password = password;
    }
    private void writeObject(ObjectOutputStream out) throws IOException{
        PutField putFields= out.putFields();
        System.out.println("原密码：" + password);
        password = "encryption";
        putFields.put("password", password);
        System.out.println("加密后的密码是 ：" + password);
        out.writeFields();
    }

    private void readObject(ObjectInputStream in) throws IOException,ClassNotFoundException{
        GetField getFields= in.readFields();
        Object objects= getFields.get("password", "");
        System.out.println("要解密的字符串是： "+ objects.toString());
        password = "pass";
    }

    public static void main(String args[]) throws IOException, ClassNotFoundException{
        ObjectOutputStream out= new ObjectOutputStream(new FileOutputStream("result.obj"));
        out.writeObject(new Test());
        out.close();

        /**
         * 序列化过程中，虚拟机会试图调用对象里的writeObject和readObject方法，进行用户自定义的序列化和反序列化
         * 用户自定义的writeObject和readObject方法允许用户控制序列化的过程，可以动态改变序列化的数值。
         */

        ObjectInputStream in = new ObjectInputStream(new FileInputStream("result.obj"));
        Test t = (Test) in.readObject();
        System.out.println("解密后的字符串是：" + t.getPassword());
        in.close();
    }
}
