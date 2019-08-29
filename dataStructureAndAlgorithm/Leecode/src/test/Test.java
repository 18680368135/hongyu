package test;

public class Test {
    public static void main(String args[]){
        String s = "456";
        StringBuffer sbu = new StringBuffer("dshu");
        sbu = new StringBuffer(sbu.substring(0,2));
        System.out.println(sbu);
        sbu.append("1");
        System.out.println(sbu);
        System.out.println(s.charAt(0)-'0');

    }
}
