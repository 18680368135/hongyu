package generic;

import java.util.Random;

class genericInterfaceRealImplement implements genericInterface<String>{

    String[] fruit = new String[] {"apple", "pear", "orange"};
    public String next(){
        Random num = new Random();
        return fruit[num.nextInt(3)];
    }
}