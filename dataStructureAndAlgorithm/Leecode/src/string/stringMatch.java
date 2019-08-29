package string;

public enum stringMatch{
    I("I",1), V("V", 5), X("X",10), L("L",50), C("C",100), D("D",500), M("M",1000),
    IV("IV",4),IX("IX",9),XL("XL",40),XC("XC",90),CD("CD",400),CM("CM",900);
    private final String key;
    private final int value;
    private stringMatch(String key, int value){
        this.key = key;
        this.value = value;
    }
    public static stringMatch getEnumByKey(String key){
        if(null == key){
            return null;
        }
        for(stringMatch temp: stringMatch.values()){
            if(temp.getKey().equals(key)){
                return temp;
            }
        }
        return null;
    }
    public String getKey(){
        return key;
    }
    public int getValue(){
        return value;
    }
}
