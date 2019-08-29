package generic;

public class genericInterfaceImplement<T> implements genericInterface<T>{
        /**
         * 未传入泛型实参时，与泛型类的定义相同，在申明类的时候，需将泛型的申明一起加到类中
         * 即： class genericInterfaceImplement<T> implement genericClass<t>
         *
         * @return
         */

        public T next(){
            return null;
        }
    }

