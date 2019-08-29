package generic;


import java.util.ArrayList;
import java.util.List;

public class genericClass<T> {
    private T key;
    public genericClass(T key){
        this.key = key;
    }

    public T getKey(){
        return this.key;
    }
    public void showKeyValue(genericClass<?> obj){
        /**
         * 使用通配符可以在逻辑上同时表示genericClass<Integer> 和genericClass<Number>父类的引用类型。
         * 通配符就是 " ？"
          */
        System.out.println("泛型测试， key value is " + obj.getKey());
    }

    public void main(String args[]){
        List<String> StringList = new ArrayList<String>();
        List<Integer> IntegerList = new ArrayList<Integer>();
        Class StringListGetClass = StringList.getClass();
        Class IntegerListGetClass = IntegerList.getClass();

        if(StringListGetClass.equals(IntegerListGetClass)){
            System.out.println("泛型测试，类型相同");

        }
        /**
         * 上面的例子，在编译之后程序会采取去泛型化的措施。也就是说Java中的泛型只在编译阶段有效。编译过程中，
         * 正确检验泛型结果后，会将泛型的相关信息擦除，并且在对象进入和离开方法的边界处添加类型检查和类型转换的方法。
         * 泛型信息不会进入运行时的阶段。
         *
         * 泛型类型在逻辑上可以看成多个不同的类型，实际上都是相同的基本类型。
         */

        genericClass<String> stringGeneric = new genericClass<String>("ajsoijf");
        genericClass<Integer> integerGeneric= new genericClass<Integer>(123456);
        System.out.println(stringGeneric.getKey());
        System.out.println(integerGeneric.getKey());
        genericClass generic1 = new genericClass("sdjioe");
        genericClass generic2 = new genericClass(1234567);
        genericClass generic3 = new genericClass(45.5689);
        genericClass generic4 = new genericClass(false);
        System.out.println(generic1.getKey().getClass());
        System.out.println(generic2.getKey().getClass());
        System.out.println(generic3.getKey().getClass());
        System.out.println(generic4.getKey().getClass());
        /**
         * 定义的泛型类一定要传入泛型类型参数吗？并不是这样，在使用泛型的时候如果传入泛型实参，则会根据传入的泛型实参做相应的限制，此时的泛型才会起到本应该起到的限制作用
         * 如果不传入泛型实参的话，在泛型类中泛型的方法或成员变量定义的类型可以为任何的类型。
         */

        genericClass<Integer> gInteger = new genericClass<Integer>(123);
        genericClass<Number> gNumber = new genericClass<Number>(789);
        showKeyValue(gNumber);
        showKeyValue(gInteger);
    }
}
