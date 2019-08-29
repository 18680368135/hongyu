package serializable;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class DeserializeDemo {
    public static void main(String args[]) throws IOException, ClassNotFoundException{
        Employee e = null;
        FileInputStream filein = new FileInputStream("/tmp/employee.ser");
        ObjectInputStream in = new ObjectInputStream(filein);
        e = (Employee) in.readObject();
        in.close();
        filein.close();
        System.out.println("Deserialized Employee ...");
        System.out.println("Name : " + e.name);
        System.out.println("address : " + e.address);
        System.out.println("SSN : " + e.SSN);
        System.out.println("number : " + e.number);
    }
}
