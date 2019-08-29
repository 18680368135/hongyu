package stringTest;

public class TestString {

    public static void main(String args[]){
    StringBuffer sb = new StringBuffer();
    sb.append("a");
    sb.append("b");
    sb.append("c");
    String str =  new String(sb);
    System.out.println(str);

    String  a = "f";
    a += "g";
    System.out.println(a);
    sb.insert(2,".");
    int i = sb.lastIndexOf(".");
    sb.deleteCharAt(i);
    System.out.println(sb.substring(1,2));
    i--;

    System.out.println(sb);
    }
}
