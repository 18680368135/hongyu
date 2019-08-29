package algorithm.backTracking;

import java.util.ArrayList;
import java.util.List;

public class PartitionSubString {
    /**
     *
     * （1）分割字符串
     * （2）需要取子串
     * （3）判断子串是否回文
     */

    List<List<String>> res = new ArrayList<>();
    public List<List<String>> partition(String s){
        nextWords(s, 0, new ArrayList<String >());
        return res;
    }

    public void nextWords(String s, int index, List<String> list){
        if(index == s.length()){
            res.add(new ArrayList<>(list));
        }
        for(int i = index; i < s.length(); i++){
            String subStr = s.substring(index,i+1);
            if(isPalindrome(subStr)){
                list.add(subStr);
                nextWords(s,i+1, list);
                list.remove(list.size()-1);
            }
        }
    }
    public boolean isPalindrome(String s){
        for(int i = 0; i <= s.length()/2; i++){
            if(s.charAt(i) != s.charAt(s.length()-1-i))
                return false;
        }
        return true;
    }

    public static void main(String args[]){
        String s = "dhaisohhfiaehohco";
        PartitionSubString psb = new PartitionSubString();
        System.out.println(psb.partition(s));
    }
}
