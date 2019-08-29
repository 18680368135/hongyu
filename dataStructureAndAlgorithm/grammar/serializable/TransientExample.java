package serializable;

import java.io.*;

public class TransientExample {
    public static void main(String args[]) throws IOException, ClassNotFoundException {

        Rectangle rectangle= new Rectangle(3,4);
        System.out.println("原始对象："+ rectangle);
        ObjectOutputStream o = new ObjectOutputStream(new FileOutputStream("rectangle"));
        o.writeObject(rectangle);

        //从流中读取对象
        ObjectInputStream in = new ObjectInputStream(new FileInputStream("rectangle"));
        Rectangle rectangle1 = (Rectangle) in.readObject();
        System.out.println("反序列化后的对象"+ rectangle1);
        rectangle1.setArea();
        System.out.println("恢复成原始对象"+rectangle1);
        in.close();
    }
}
