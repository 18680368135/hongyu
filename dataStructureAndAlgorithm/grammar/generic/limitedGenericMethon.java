package generic;
public class limitedGenericMethon {
    /**
     *
     * @param x
     * @param y
     * @param z
     * @param <E> 继承可以对比的数据类型，x y z 可以为任何类型的数据，只要可以比较就行
     * @return
     */
    public static <E extends Comparable<E>> E maxComparable(E x, E y, E z){
        E max = x;
        if(y.compareTo(max) > 0){
            max = y;
        }
        if(z.compareTo(max) > 0){
            max = z;
        }
        return max;
    }

    public static void main(String args[]){
        System.out.println(maxComparable(1.5, 8.6, 4.8));
        System.out.println(maxComparable("pear", "apple", "orange"));
        System.out.println(maxComparable(123, 456, 789));
    }
}
