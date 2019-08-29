package generic;

public class genericMethod {

    /**
     * 泛型方法
     */
    public static <E> void print_number(E[] inputarray){
        for(E element: inputarray){
            System.out.print(element + " ");
        }
        System.out.println();
    }

    public static void main(String args[]){
        Double[] double_array = {1.1,2.2,4.4,5.5,6.6,7.7,9.9};
        Integer[] int_array = {1,2,3,4,5,6,7,8,9};
        Character[] char_array = {'a','b','c','d','e'};
        print_number(double_array);
        print_number(int_array);
        print_number(char_array);

    }
}
