package algorithm.backTracking;

import java.util.*;

public class LetterCombinations {
    public List<String> readDigitString(String digits){
        Map<Character, String> dict = new HashMap<>();
        dict.put('0', "");
        dict.put('1', "");
        dict.put('2', "abc");
        dict.put('3', "def");
        dict.put('4', "ghi");
        dict.put('5', "jkl");
        dict.put('6', "mno");
        dict.put('7', "pqrs");
        dict.put('8', "tuv");
        dict.put('9', "wxyz");
        char[] charArr = digits.toCharArray();
        List<String> str = new ArrayList<String>();
        for(Character a:charArr) {
            str.add(dict.get(a));
        }
        return str;


    }
    public List<String> letterCombinations1(String digits){
        /**
         输入："23" 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
         说明: 尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。
         **************************************/
         /************************************** *
          想法：
          * 1. 循环0~length-1
          * 将原数组中的值加上当前值得可能性
          * 比如第一个数字2,对应的就是abc，这时候数组中的值就是[a,b,c]
          * 轮到下一个数字3，对应的是def,就把a拼上d/e/f
          * 返回数组
          * 2. 回溯法
          * 把每个数字当作递归的一层，每一层中先枚举一个字母，
          * 递归进入下一层，再删除这个字母，回到上一个状态，枚举下一个字母。
          * 递归结束标志是递归了digits.lengtgh层，即字母组合长度等于digits长度，
          * 递归结束得到一个符合的字母组合，加入list。等于是在循环中套递归
          * 我的做法
          * 时间复杂度：n2
          * 空间复杂度：n
          * 代码执行过程：
          总结：
          没有用递归的方式，直接使用迭代的方法

         */
        List<String> lis = readDigitString(digits);
        List<String> result = new ArrayList<String>();
        result.add("");
        if(digits.length() == 0) return Collections.EMPTY_LIST;
        for(int i = 0; i < lis.size(); i++){
            List<String> temList = new ArrayList<String>();
            for(String str: result) {
                for (Character ch : lis.get(i).toCharArray()) {
                    String temArr = str + ch;
                    temList.add(temArr);
                }
            }
            result = temList;
        }
        return result;
    }
    public List<String> letterCombinations(String digits){
        List<String> result = new ArrayList<String>();
        String oneRes = "";
        if(digits.equals(""))return result;
        String[] digMapStr = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        // 将对应的整数字符串转化为整数数组
        int[] digInt = new int[digits.length()];
        for(int i = 0; i < digits.length(); i++) digInt[i] = digits.charAt(i) - '0';
        combi(digInt, 0, digMapStr, result, oneRes);

        return result;
    }
    public void combi(int[] digInt, int n, String[] digMapStr,
                      List<String> result, String oneRes){
        if(n == digInt.length){
            result.add(oneRes);
            return;
        }

        for(int j = 0; j < digMapStr[digInt[n]].length(); j++){
            oneRes = oneRes + digMapStr[digInt[n]].charAt(j);
            //System.out.println("before:"+ oneRes);
            combi(digInt, n+1, digMapStr, result, oneRes);
            //System.out.println("after: "+oneRes);
            oneRes = oneRes.substring(0, oneRes.length()-1);
        }
    }

    public static void main(String args[]){
        List<String> testList = new ArrayList<String>();
        testList.add("");
        testList.add("23");
        testList.add("56");
        testList.add("89");
        testList.add("489");
        LetterCombinations lc = new LetterCombinations();
        for(String digits: testList){
            System.out.println(lc.letterCombinations1(digits));
        }

    }
}
