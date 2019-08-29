package serializable;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class SerializeDemo {
    public static void main(String args[]) throws IOException{
        Employee e = new Employee();
        e.name = "Reyan Ali";
        e.address = "Phokka Kuan, Ambehta Peer";
        e.SSN = 11122333;
        e.number = 101;
        FileOutputStream fileout = new FileOutputStream("/tmp/employee.ser", true);
        ObjectOutputStream out= new ObjectOutputStream(fileout);
        out.writeObject(e);
        out.close();
        fileout.close();
        System.out.println("Serialized data is in /tmp/employee.ser");
    }
}
