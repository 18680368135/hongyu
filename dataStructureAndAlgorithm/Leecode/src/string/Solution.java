package string;

import java.util.*;

public class Solution {
    /**
     * 1 、讲罗马数字转化成阿拉伯数字，分别使用枚举、映射、数组
     * 2 、编写一个函数来查找字符串数组中的最长公共前缀
     * 3 、给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效
     * 4 、实现strStr()函数。
     * 5 、报数
     * 6 、最后一个单词的长度
     * 7 、二进制求和
     * 8 、反转字符串
     * 9 、反转字符串中的元音字母
     * 10 、赎金信
     * 11 、字符串中第一个唯一的字符
     * 12 、字符串相加
     * 13 、字符串中的单词数
     * 14 、压缩字符串
     * 15 、重复的字符串
     * 16 、检查大写字母
     * 17 、最长特殊序列I
     * 18 、反转字符串
     * 19 、最长特殊序列II
     * 20 、学生出勤记录1
     * 21 、反转字符串中的单词III
     * 22 、根据二叉树创建字符串
     * 23 、机器人能否返回原点
     * 24 、验证回文串II
     * 25 、重复叠加字符串匹配
     * 26 、计数二进制子串
     * 27 、转换成小写字母
     * 28 、旋转数字
     * 29 、唯一摩尔斯密码
     * 30 、最常见的单词
     * 31 、山羊拉丁文
     * 32 、亲密字符串
     * 33 、特殊等价字符数组
     * 34 、字符串的最大公因子
     * 35 、重新配列日志文件
     * 36 、独特的电子邮件地址
     * 37 、长按键入
     * 38 、仅仅反转字母
     **/
    //枚举
    public int romanToInt(String s){
        if(s.isEmpty()) return 0;
        int sum = 0,length = s.length();
        for(int i = 0; i < length; i++){
            String str;
            if(i < length-1){
                if(s.charAt(i)!=s.charAt(i+1)){
                    str = s.substring(i,i+2);
                    if(stringMatch.getEnumByKey(str) != null){
                        sum += stringMatch.getEnumByKey(str).getValue();
                        i++;
                        continue;
                    }
                }
            }
            str = s.substring(i,i+1);
            sum += stringMatch.getEnumByKey(str).getValue();
        }
        return sum;
    }
    //映射
    public int romanToInt1(String s) {
        if(s.isEmpty()) return 0;
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        int sum = 0,length = s.length();
        for(int i = 0; i < length; i++){
            char c = s.charAt(i);
            if(i < length-1 && map.get(c) < map.get(s.charAt(i+1))){
                sum += map.get(s.charAt(i+1)) - map.get(c);
                i++;
            }else {
                sum += map.get(c);
            }
        }
        return sum;
    }
    //使用数组
    static int[] val = new int[]{1000, 500, 100, 50, 10, 5, 1};
    static char[] c = new char[]{'M', 'D', 'C', 'L', 'X', 'V', 'I'};
    public int romanToInt2(String s){
        int sum = 0, lenght = s.length(), t = 0;
        for(int i = 0; i < lenght; i++){
            while (s.charAt(i) != c[t])
                t++;
            if(i+1 < lenght && ((t>0 && s.charAt(i+1) == c[t-1]) ||
                            (t>1 && s.charAt(i+1) == c[t-2]))){
                sum -= val[t];
            }else {
                sum += val[t];
            }
            t = 0;
        }
        return sum;
    }

    /**
     * 编写一个函数来查找字符串数组中的最长公共前缀
     * @param strs
     * @return
     */

    public String longestCommonPrefix(String[] strs) {
        //比较相邻的两个字符串的最长公共前缀，最终保留最短的公共前缀
        if(strs.length == 0)return "";
        int size = strs.length, length = strs[0].length(), t = 0;
        String str = strs[0];
        for(int i = 0; i < size-1; i++){
            int tempLen = 0;
            while (!strs[i].isEmpty() &&
                    !strs[i+1].isEmpty() &&
                    strs[i].charAt(t) == strs[i+1].charAt(t)){
                tempLen++;
                t++;
                if(t == strs[i].length() || t == strs[i+1].length())
                    break;
            }
            if(length > tempLen){
                length = tempLen;
                str = strs[i].substring(0,length);
            }
            t = 0;
        }
        return str;
    }
    public String longestCommonPrefix1(String[] strs) {
        if(strs.length == 0)
            return "";
        int len = strs[0].length(), size = strs.length;
        String str = strs[0];
        for(int i = 0; i < len; i++){
            char c = str.charAt(i);
            for(int j = 0; j < size - 1; j++){
                if( i == strs[j+1].length() || c != strs[j+1].charAt(i))
                    return str.substring(0,i);
            }
        }
        return str;
    }
    String[] strr;
    public String longestCommonPrefix2(String[] strs) {
        int min = 10000; int flag = 0;
        if(strs.length == 0){
            return "";
        }
        for(int i =0; i< strs.length; i++){
            if(strs[i].length() < min){
                flag = i;
                min = strs[i].length();
            }
        }
        this.strr = strs;

        for(int i = strs[flag].length(); i>=0 ; i--){
            String temp = strs[flag].substring(0,i);
            if(pan(temp)){
                return temp;
            }
        }
        return "";
    }

    public boolean pan(String temp ){
        for(int i =0 ; i < strr.length; i++){
            if(!strr[i].startsWith(temp)){
                return false;
            }
        }
        return true;
    }

    /**
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
     有效字符串需满足：

     左括号必须用相同类型的右括号闭合。
     左括号必须以正确的顺序闭合。

     注意空字符串可被认为是有效字符串。

     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s.isEmpty()) return true;
        Map<Character, Character> map = new HashMap<>();
        map.put('[', ']');
        map.put('{', '}');
        map.put('(', ')');
        Stack<Character> stack = new Stack<>();
        int length = s.length();
        char[] chararray = s.toCharArray();
        for (char ch : chararray) {
            if (ch == '{' || ch == '(' || ch == '[')
            {
                stack.push(ch);
                continue;
            }
        if (!stack.isEmpty() && ch == map.get(stack.pop()))continue;
        return false;
    }
    return stack.isEmpty();
    }

    /**
     * 写成子函数判断括号是否匹配
     */
    public boolean isValid1(String s) {
        Stack<Character> stack = new Stack<>();
        char[] charArray = s.toCharArray();
        for(char ch :charArray){
            if(stack.isEmpty()){
                stack.push(ch);
            }else if(isMatch(stack.peek(), ch)){
                stack.pop();
            }else {
                stack.push(ch);
            }
        }
        if(stack.isEmpty())return true;
        return false;
    }
    public boolean isMatch(char a , char b){
        return (a == '(' && b == ')') || (a == '[' && b == ']') || (a == '{' && b == '}');
    }

    /**
     * 给定一个 haystack 字符串和一个 needle 字符串，
     * 在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
     * 如果不存在，则返回  -1。
     * @param haystack
     * @param needle
     * @return
     */
    public int strStr(String haystack, String needle) {
        if(needle.length() == 0)return 0;
        int len = haystack.length(), llen = needle.length(),j = 0;
        for(int i = 0; i < len; i++) {
            if (len>0) {
                if (haystack.charAt(i) == needle.charAt(j))
                j++;
                else {
                    i -= j;
                    j = 0;
                     }
            }
            if (j == llen) {
                return i - llen + 1;
            }
        }

        return -1;
    }
    public int strStr1(String haystack, String needle) {
        return haystack.indexOf(needle);
    }

    public int strStr2(String haystack, String needle) {
        if(needle.length() == 0)return 0;
        String temp;
        int len = haystack.length(), nLen = needle.length();
        for(int i = 0; i < len-nLen+1; i++){
            temp = haystack.substring(i, i+nLen);
            if(temp.equals(needle)){
                return i;
            }
        }
        return -1;
    }

    /**
     *
     报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：

     1.     1
     2.     11
     3.     21
     4.     1211
     5.     111221

     1 被读作  "one 1"  ("一个一") , 即 11。
     11 被读作 "two 1s" ("两个一"）, 即 21。
     21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。
     给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。
     注意：整数顺序将表示为一个字符串。
     */
    public String countAndSay(int n) {
        return sayCount(n);
    }
    public String sayCount(int n){
        StringBuffer sb= new StringBuffer();
        if(n==1) return "1";
        String temp = sayCount(n-1);
        char ch = temp.charAt(0);
        int count = 1;
        for(int i = 1; i < temp.length(); i++){
            if(ch != temp.charAt(i)){
                sb.append(Integer.toString(count));
                sb.append(ch);
                ch = temp.charAt(i);
                count = 1;
            }else {
                count++;
            }
        }
        sb.append(Integer.toString(count));
        sb.append(ch);
        return sb.toString();
    }

    /**
     * 给定一个仅包含大小写字母和空格的字符串，返回其最后一个单词的长度
     * @param s
     * @return
     */
    public int lengthOfLastWord(String s) {
        int start = 0, end = 0;
        boolean mark = false;
        int len = 0;
        int length = 0;
        for(char ch: s.toCharArray()){
            if(ch !=  ' ' && !mark){
                start = end;
                mark = true;
            }
            if(ch == ' ' && mark){
                len = end - start;
                mark = false;
            }
            end++;
            length++;
            if(length == s.length() && mark){
                len = end-start;
            }
        }
        return len;
    }

    /**
     * 从后往前，找最后一个单词
     */
    public int lengthOfLastWord1(String s) {
        int len = s.length(), sum = 0;
        boolean mark = false;
        for(int i = len-1; i >= 0; i--){
            if(s.charAt(i) == ' ' && mark){
                break;
            }
            if(s.charAt(i) != ' '){
                mark = true;
                sum++;
            }
        }
        return sum;
    }

    /**
     * 给定两个二进制字符串，返回他们的和（用二进制表示）
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        int pre = 0;
        if(b.length() > a.length()){
            String temp = a;
            a = b;
            b = temp;
        }
        int lena = a.length()-1, lenb = b.length()-1;
        StringBuilder sb = new StringBuilder();
        while (lena >= 0){
            int numb = 0;
            int numa = a.charAt(lena) - '0';
            if(lenb >= 0){
                numb = b.charAt(lenb) - '0';
            }
            if(numa + numb + pre >= 2){
                sb.append((numa + numb + pre) %2);
                pre = 1;
            }else {
                sb.append(numa + numb + pre);
                pre = 0;
            }
            lena--;
            lenb--;
        }
        if(pre == 1)
            sb.append(pre);
        return sb.reverse().toString();

    }
    public String addBinary1(String a, String b) {
        StringBuilder res = new StringBuilder();
        int lena = a.length()-1, lenb = b.length()-1,pre = 0;
        while(lena >-1 || lenb > -1){
            int sum = pre;
            sum += lena > -1 ? a.charAt(lena--)-'0': 0;
            sum += lenb > -1 ? b.charAt(lenb--)-'0': 0;
            res.append(sum%2);
            pre = sum/2;

        }
        res.append(pre == 0?"":pre);
        return res.reverse().toString();
    }

    /**
     * 编写一个函数，其作用是将输入的字符串反转过来。
     * 输入字符串以字符数组 char[] 的形式给出。
     不要给另外的数组分配额外的空间，你必须原地修改输入数组、
     使用 O(1) 的额外空间解决这一问题。

     * @param s
     */
    public void reverseString(char[] s) {
        if(s.length == 0)return;
        for(int i = 0; i < s.length/2; i++){
            char temp = s[i];
            s[i] = s[s.length-1-i];
            s[s.length-1-i] = temp;
        }
    }
    /**
     * 编写一个函数，以字符串作为输入，反转该字符串中的元音字母。
     */
    public String reverseVowels(String s) {
        if(s == null) return s;
        char[] res = s.toCharArray();
        int i = 0, j = s.length()-1;
        while(i < j){
            while(i < j && !isValidVowels(res[i])) i++;
            while(i < j && !isValidVowels(res[j])) j--;
            if(i < j){
                char temp = res[i];
                res[i] = res[j];
                res[j] = temp;
                i++;
                j--;
            }
            }
        return String.valueOf(res);
    }
    public boolean isValidVowels(char a){
        return a=='a' || a=='i' || a=='o' || a=='e' || a=='u' ||
                a=='A' || a=='I' || a=='O' || a=='E' || a=='U' ;
    }

    /**
     * 给定一个赎金信字符串和一个杂志字符串，
     * 判断第一个字符串能不能由第二个字符串里面的字符构成，
     * 如果可以构成，返回true，否则返回false
     */

    /**
     * 解错，理解错题意了，并不是字符串的完全匹配，而是字符串的重构，采用KMP算法并不合适，更改策略
     *
     * 采用KMP算法进行字符串的模式匹配，不过关键是提前计算好next数组的值
     * @param ransomNote
     * @param magazine
     * @return
     */
    public boolean canConstruct(String ransomNote, String magazine) {
        if(ransomNote == null)return true;
        int lena = ransomNote.length(), lenb = magazine.length();
        if(lena > lenb) return false;
        int[] next = new int[lena];
        getNext(ransomNote, next);
        int i = 0,j = 0;
        while(i < lena && j < lenb){
            if(i == -1 || ransomNote.charAt(i) == magazine.charAt(j)){
                i++;
                j++;
            }else {
                i = next[i];
            }
        }
        if(i == lena) return true;
        else return false;
    }

    public void getNext(String ransomNote, int next[]){
        int len = ransomNote.length()-1;
        next[0] = -1;
        int i = 0;
        int k = -1;
        while(i < len){
            if(k == -1 || ransomNote.charAt(i) == ransomNote.charAt(k)){
                i++;
                k++;
                next[i] = k;
            }else {
                k = next[k];
            }
        }
    }
    public boolean canConstruct1(String ransomNote, String magazine) {
        if(ransomNote.length() == 0)return true;
        List<Character> list = new LinkedList<>();
        for(char cha: ransomNote.toCharArray()){
            list.add(cha);
        }
        for(char chb: magazine.toCharArray()){
            if(list.contains(chb))
                list.remove(list.indexOf(chb));
            if(list.isEmpty())
                return true;
        }
        return false;
    }

    /**
     * 数组索引结合字符,时间最短
     */
    public boolean canConstruct2(String ransomNote, String magazine) {
        int n = 0;
        int lena = ransomNote.length(), lenb = magazine.length();
        if(lena > lenb) return false;
        int[] mark = new int[28];
        for(char ch : ransomNote.toCharArray()){
            mark[ch-97]++;
            n++;
        }
        for(char ch : magazine.toCharArray()){
            if(mark[ch-97] > 0) {
                mark[ch-97]--;
                n--;
            }
        }
        return n <= 0;
    }

    /**
     * 采用hashMap 来做，key-value----> character- Integer
     * @param ransomNote
     * @param magazine
     * @return
     */
    public boolean canConstruct3(String ransomNote, String magazine) {
        if(ransomNote.length() > magazine.length())return false;
        Map<Character, Integer> map= new HashMap<>();
        for(char ch: ransomNote.toCharArray()){
            if(map.containsKey(ch)){
                map.put(ch, map.get(ch)+1);
            }else {
                map.put(ch, 1);
            }
        }
        for(char ch: magazine.toCharArray()){
            if(map.containsKey(ch)){
                map.put(ch, map.get(ch)-1);
                if(map.get(ch) == 0){
                    map.remove(ch);
                }
            }
        }
        if(map.isEmpty())return true;
        else return false;
    }

    /**
     * 给定一个字符串s，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回-1.
     * @param s
     * @return
     */
    public int firstUniqChar(String s) {
        if(s == null || s.length() == 0) return -1;
        short[] count = new short[27];
        for(char ch: s.toCharArray()){
            count[ch-'a']++;
        }
        for(int i = 0; i < s.length(); i++){
            if(count[s.charAt(i) - 'a'] == 1)
                return i;
        }
        return -1;

    }
    public String addStrings(String num1, String num2){
        int lena = num1.length()-1, lenb = num2.length()-1, pre = 0, sum;
        StringBuilder res = new StringBuilder();
        while(lena >= 0 || lenb >= 0){
            sum = pre;
            sum += lena >= 0 ? num1.charAt(lena--)-'0': 0;
            sum += lenb >= 0 ? num2.charAt(lenb--)-'0': 0;
            res.append(sum%10);
            pre = sum/10;
        }
        if(pre == 1)
            res.append(pre);
        return res.reverse().toString();
    }

    /**
     * 统计字符串中的单词个数，这里的单词值得是连续的不是空格的字符
     * @param s
     * @return
     */
    public int countSegments(String s){
        if(s == null || s.length() == 0)return 0;
        s = s.trim();
        if(s.length() == 0)return 0;
        int count = 1;
        boolean mark = false;
        for(char ch : s.toCharArray()){
            if(ch == ' '&& !mark){
                mark = true;
            }
            if(ch != ' ' && mark){
                mark = false;
                count++;
            }
        }
        return count;
    }

    /**
     * 不用trim 消除字符串两端的空格
     * @param s
     * @return
     */
    public int countSegments1(String s) {
        int i = 0, len = s.length(), pre = 0,total = 0;
        while(i < len){
            if(pre == 0 && s.charAt(i) != ' '){
                pre = 1;
                total++;
            }if(s.charAt(i) == ' ') pre = 0;
            i++;
        }
        return total;
    }

    /**
     * 给定一组字符，使用原地算法将其压缩
     * 压缩后的长度必须始终小于或等于原数组的长度
     * 数组的每个元素应该是长度为1的字符（不是int整数类型）
     * 完成原地修改输入数组后，返回数组的新长度
     * @param chars
     * @return
     */
    public int compress(char[] chars){
        int i = 0;
        int start = 0;
        int end = 0;
        while(start < chars.length){
            while(end < chars.length && chars[start] == chars[end]){
                end++;
            }
            chars[i++] = chars[start];
            if(end - start >1){
                String  str = Integer.toString(end-start);
                for(int j = 0; j < str.length(); j++){
                    chars[i++] = str.charAt(j);
                }
            }
            start = end;
        }
        return i;
    }

    /**
     * 给定一个非空的字符串，判断他是否可以由它的一个字串重复多次构成。
     * 给定的字符串只含有小写英文字母，并且长度不超过10000.
     * 执行用时 :574 ms, 在所有 Java 提交中击败了5.51% 的用户
     * 内存消耗 :368 MB, 在所有 Java 提交中击败了7.13%的用户
     * @param s
     * @return
     */
    public boolean repeatedSubstringPattern(String s) {
        if(s.length() == 1)return true;
        int len = s.length(), i = 1;
        while(i < len/2){
            String str = s.substring(0, i);
            if(len % str.length() != 0){
                i++;
                continue;
            }
            int index = i;
            int indexNext = i * 2;
            while(str.equals(s.substring(index,indexNext))){
                index += i;
                indexNext += i;
                if(index == len)return true;
            }
            i++;

        }
        return false;
    }
    public boolean repeatedSubstringPattern1(String s){
        int len = s.length();
        int[] next = new int[len+1];
        next[0] = -1;
        int j = 0;
        int k = -1;
        while(j < len){
            if(k == -1 || s.charAt(k) == s.charAt(j)){
                k++;
                j++;
                next[j] = k;
            }else {
                k = next[k];
            }
        }
        return next[len] != 0 && len%(len -next[len]) == 0;
    }

    public boolean detectCapitalUse(String word){
        int len = word.length(),i = 0;
        if(len == 0) return false;
        char ch = word.charAt(i);
        if(ch >= 'a' && ch <= 'z'){
            while(++i < len)
            {
                ch = word.charAt(i);
                if(ch < 'a' || ch > 'z'){
                    return false;
                }
            }
        }else if(ch >= 'A' && ch <= 'Z'){
            if (++i < len){
                ch = word.charAt(i);
                if(ch >= 'a' && ch <= 'z') {
                    while (++i < len) {
                        ch = word.charAt(i);
                        if (ch < 'a' || ch > 'z') {
                            return false;
                        }
                    }
                }
                if(ch >= 'A' && ch <= 'Z') {
                    while (++i < len) {
                        ch = word.charAt(i);
                        if (ch < 'A' || ch > 'Z') {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    /**
     * 第一个字母大小写无所谓，只要后面的全部大写或全部小写就行
     * @param word
     * @return
     */
    public boolean detectCapitalUse1(String word) {
        int len = word.length(), i = 0;
        if(len == 0)return false;
        char ch = word.charAt(i);
        while (i < len && ch <= 'z' && ch >= 'a'){
            if(word.charAt(i) < 'a'|| word.charAt(i++) > 'z') return false;
        }
        if( i < len && --len > 0){
            word = word.substring(1);
            ch = word.charAt(i);
        }
        while (i< len && ch >= 'A' && ch <= 'Z'){
            if(word.charAt(i) < 'A'|| word.charAt(i++) > 'Z') return false;
        }
        while (i< len && ch >= 'a' && ch <= 'z'){
            if(word.charAt(i) < 'a'|| word.charAt(i++) > 'z') return false;
        }
        return true;
    }

    /**
     * 给定两个字符串，你需要从这两个字符串中找出最长的特殊序列，最长特殊序列定义如下：
     * 该序列为某字符串独有的最长子序列（既不能是其它字符串的子序列）
     *
     * 子序列可以通过删去字符串中的某些字符发现，但不能改变剩余字符的相对顺序。
     * 空序列为所有字符串的子序列，任何字符串为其自身的子序列。
     *
     * 输入两个字符串，输出最长特殊序列的长度，如果不存在，则返回-1.
     * @param a
     * @param b
     * @return
     */
    public int findLUSlength(String a, String b){
        if(a.equals(b))return -1;
        return (a.length() > b.length())? a.length() : b.length();
    }

    /**
     * 反转字符串
     * @param s
     * @param k
     * @return
     */
    public String reverseStr(String s, int k){
        char[] cha = s.toCharArray();
        int len = s.length();
        int l = len%(2*k)==0?len/(2*k):(len/(2*k) +1);
        int left,right;
        for(int j = 0; j< l; j++) {
            left = 2 * j * k;
            if ((2 * j + 1) * k > len) {
                right = len-1;
                reverse(left, right,cha);
            } else{
                right = 2*j*k+k-1;
                reverse(left,right,cha);
            }
        }
        return String.valueOf(cha);
    }
    public void reverse(int left, int right, char[] cha){
        while(left <= right) {
            char temp = cha[left];
            cha[left] = cha[right];
            cha[right] = temp;
            left++;
            right--;
        }
    }

    /**
     * 给定字符串列表，你需要从它们中找出最长的特殊序列。
     * 最长特殊序列定义如下：该序列为某字符串独有的最长序列（即不能是其它字符串的子序列）
     *
     * 子序列可以通过删去字符串中的某些字符实现，但不能改变剩余字符的相对顺序。
     * 空序列为所有字符串的子序列，任何字符串为其自身的子序列。
     *
     * 输入将是一个字符串列表，输出是最长特殊序列的长度。
     * 如果特殊序列不存在，返回-1.
     * @param strs
     * @return
     */
    public int findLUSlength(String[] strs){
        return 0;
    }

    public boolean checkRecord(String s){
        int countAbsent = 0, i = 0;
        char pre = ' ', pos = ' ';
        while(i < s.length()){
            char ch = s.charAt(i++);
            if(ch == 'A')countAbsent++;
            if(i==0)  {pre = ch; continue;}
            if(i==1)  {pos = ch; continue;}

            if(countAbsent >1 || (pre == 'L'&& pos == 'L' && ch == 'L'))
                return false;
            pre = pos;
            pos = ch;
        }
        return true;
    }

    /**
     *利用缺勤和到场清空迟到
     */

    public boolean checkRecord1(String s){
        int absent = 0, late = 0;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == 'A'){
                late = 0;
                if(++absent > 1)return false;
            }else if(s.charAt(i) == 'L'){
                if(++late > 2)return false;
            }else {
                late++;
            }
        }
        return true;
    }

    /**
     * 给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
     * @param s
     * @return
     */
    public String reverseWords(String s){
        int left = 0, right = 0, i = 0, len = s.length() ;
        boolean markl = false;
        char[] cha = s.toCharArray();
        while(i < len){
            if(cha[i] != ' ' && !markl)
            {
                left = i;
                markl = true;
            }
            if(cha[i] == ' ' || i == len-1){
                right = i;
            }
            if(left < right){
                right = i == len-1?right: right-1;
                reverse(left,right,cha);
                markl = false;
                left = i;
                right = i;
            }
            i++;
        }
        return String.valueOf(cha);
    }

    /**
     * 先利用字符串切分成数组，然后利用StringBuilder对其每一项项进行反转
     * @param s
     * @return
     */
    public String reverseWords1(String s){
        String[] str = s.split(" ");
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < str.length; i++){
            StringBuilder sb = new StringBuilder(str[i]);
            res.append(sb.reverse());
            if(i != str.length-1)res.append(" ");
        }
        return res.toString();
    }

    /**
     *你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
     * 空节点则用一对空括号表示。
     * 而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。
     */
    public String tree2str(TreeNode t){

        StringBuilder res = new StringBuilder();
        if(t == null)return res.toString();
        find(t, res);
        return res.toString();
    }

    public void find(TreeNode t, StringBuilder res){
        if (t == null) {
            return;
        }
        res.append(t.val);
        if(t.left!=null){
            res.append('(');
            find(t.left,res);
            res.append(')');
        }else {
            if (t.right != null) {
                res.append("()");
            }
        }
        if(t.right != null){
            res.append('(');
            find(t.right,res);
            res.append(')');
        }
    }

    /**
     *在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 (0, 0) 处结束。

     移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。
     如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。

     注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。
     */
    public boolean judgeCircle(String moves){

       HashMap<Character, int[]>  map = new HashMap<>();
       map.put('R', new int[]{1,0});
       map.put('L', new int[]{-1,0});
       map.put('U', new int[]{0, 1});
       map.put('D', new int[]{0, -1});
       int a = 0,b = 0;
       for(char ch : moves.toCharArray()) {
           int[] c = map.get(ch);
           a += c[0];
           b += c[1];

       }
//       char[] chars = new char[]{'R', 'L', 'U', 'D'};
//       int[][] move = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
//       int i = 0,one = 0, two = 0;
//       for(char ch : moves.toCharArray()) {
//           while(ch != chars[i])i++;
//           one += move[i][0];
//           two += move[i][1];
//           i = 0;
//       }
       return a == 0 && b == 0;
    }

    /**
     * 给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
     * 动态规划
     * 执行用时 :7 ms, 在所有 Java 提交中击败了99.84% 的用户
       内存消耗 :38.2 MB, 在所有 Java 提交中击败了92.92%的用户
     * @param s
     * @return
     */
    public boolean validPalindrome(String s){
        return palindrome(s,0, 0, s.length()-1);
    }
    public boolean palindrome(String res, int count, int left, int right){
        if(count >1)return false;
        while(left < right){
            if(res.charAt(left) != res.charAt(right)){
                return palindrome(res, count+1,left+1, right)
                        || palindrome(res,count+1,left,right-1) ;
            }
            left++;
            right--;
        }
        return true;
    }

    /**
     * 给定两个字符串 A 和 B, 寻找重复叠加字符串A的最小次数，使得字符串B成为叠加后的字符串A的子串，
     * 如果不存在则返回 -1。
     *
     * 关键的终止长度在于 2*A+B
     * @param A
     * @param B
     * @return
     */
    public int repeatedStringMatch(String A, String B){
        int i , k=0, j = 0, count = 1;
        int lena = A.length();
        int lenb = B.length();
        while(k < lena){
            //寻找B在A中的初始位置i
            if(A.charAt(k) != B.charAt(j)){
                k++;
            }else
                {
                    i = k;
                    while(++j < lenb){
                        if(++i == lena){
                            i=0;
                            count++;
                        }
                    if(A.charAt(i) != B.charAt(j))break;
                    }
                    if(j == lenb)return count;
                    count = 1;
                    k++;
                    j=0;

                }
            if(k == lena) return -1;
        }

        return -1;

    }

    /**
     * 换一种思路，迭代的总长度不会超过 2A+B,
     * 存在子串的情况下需要重复A=2+B/A次
     * @param A
     * @param B
     * @return
     */
    public int repeatedStringMatch1(String A, String B){
        int maxCount = 2+B.length()/A.length();
        int i = 1;
        StringBuilder res = new StringBuilder();
        while(i <= maxCount){
            if(res.lastIndexOf(B) >-1)return i;
            else {
                res.append(A);
                i++;
            }
        }
        return -1;
    }

    /**
     * 采用KMP算法，匹配字符串
     * @param A
     * @param B
     * @return
     */
    public int repeatedStringMatch2(String A, String B){
        int maxCount = 2 + B.length()/A.length();
        int i = 1;
        StringBuilder res= new StringBuilder(A);
        int[] next = new int[B.length()];
        next(next, B);
        while(i <= maxCount){
            if(kmp(res.toString(),next,B))return i;
            else {
                i++;
                res.append(A);
            }
        }

        return -1;
    }
    public boolean kmp(String str,int next[], String B){
        int i = 0, j = 0;
        while (i < str.length() && j < B.length()){
            if(j == -1 || str.charAt(i) == B.charAt(j)){
                i++;
                j++;
            }else{
                j = next[j];
            }
        }
        return j == B.length();

    }
    public void next(int[] next, String B){
        int j = 0;
        int k = -1;
        next[0] = -1;
        while(j < B.length()-1){
            if(k == -1 || B.charAt(j) == B.charAt(k)){
                ++j;
                ++k;
                next[j] = k;
            }else{
                k = next[k];
            }
        }

    }

    /**
     * 给定一个字符串s,计算具有相同数量0和1的非空（连续）子字符串的数量，
     * 并且这些子字符串中的所有0和所有1都是组合在一起的。
     *
     * 重复出现的子串要计算它们出现的次数。
     * @param s
     * @return
     */
    public int countBinarySubstrings(String s){
        int count = 0,count0 = 0, count1,cur = 0;
        int len = s.length() ;
        while (cur < len){
            while (cur < len && s.charAt(count0) == s.charAt(cur))cur++;
            if (cur >= len) break;
            count1 = cur;
            while (cur < len && s.charAt(count1) == s.charAt(cur))cur++;
            count += count1-count0 < cur - count1?count1-count0:cur - count1;
            count0 = count1;
        }
        return count;
    }
    public int countBinarySubstrings1(String s){
        int pre = 0,cur = 0, count=0;
        int len = s.length() ;
        for(int i = 0; i < len-1; i++){
            if(s.charAt(i) == s.charAt(i+1))
            {
                cur++;
            }else{
                pre = cur;
                cur = 0;
            }
            if(pre >= cur){
                count++;
            }
        }
        return count;
    }

    /**
     * 将带有大写字母的字符串转化为小写字母的字符串
     * @param str
     * @return
     */
    public String toLowerCase(String str){
        char[] res = str.toCharArray();
        for(int i = 0; i < res.length; i++){
            if(res[i] >= 'A' && res[i] <= 'Z')
                res[i] = (char) (res[i] + 32);
        }
        return new String(res);
    }
    public int rotatedDigits(int N){
        int count = 0;
        Map<Character,Character> map = new HashMap<>();
        map.put('8','8');
        map.put('1','1');
        map.put('0','0');
        map.put('2','5');
        map.put('5','2');
        map.put('6','9');
        map.put('9','6');
        for(int i = 1; i <= N; i++){
            Vector<Character> vec = getNum(i);
            if(vec.size() == 1 && null != map.get(vec.get(0)) && map.get(vec.get(0)) != vec.get(0)){
                count++;
            }else{
                boolean mark = false;
                for(int j = 0; j < vec.size(); j++){
                    if(null == map.get(vec.get(j))) {
                        mark = false;
                        break;
                    }
                    if(vec.get(j) != map.get(vec.get(j)))
                        mark = true;
                }
                if(mark)count++;
            }
        }
        return count;
    }
    public Vector getNum(int N){
        Vector<Character> vec= new Vector<>();
        while(N/10 >0){
            vec.add((char)(N%10 + '0'));
            N /= 10;
        }
        vec.add((char)(N + '0'));
        return vec;
    }

    public int rotatedDigits1(int N){
        int count = 0;
        boolean mark;
        for(int i = 1; i <= N; i++){
            Vector<Character> vec = getNum(i);
            mark = false;
            for(int j = 0; j < vec.size(); j++){
                if(vec.get(j) == '3' || vec.get(j) == '4' || vec.get(j) == '7') {
                    mark = false;
                    break;
                }
                if(vec.get(j) == '2' || vec.get(j) == '5'|| vec.get(j) == '6'|| vec.get(j) == '9')
                    mark = true;
            }
            if(mark)count++;
        }
        return count;
    }
    public int rotatedDigits2(int N){
        int count = 0;
        boolean mark;
        while(N > 0){
            if(N %10 == 3 || N %10 == 4 || N %10 == 7)
            {
                N--;
                continue;
            }
            String str  = String.valueOf(N);
            int len = str.length();
            mark = false;
            for(int j = 0; j < len; j++){
                char ch = str.charAt(j);
                if(ch == '3' || ch == '4' || ch == '7') {
                    mark = false;
                    break;
                }
                if(ch == '2' || ch == '5'|| ch == '6'|| ch == '9')
                    mark = true;
            }
            if(mark)count++;
            N--;
        }
        return count;
    }
    public int rotatedDigits3(int N){
        int count = 0,mod;
        boolean mark;
        while (N > 0) {
            int num = N;
            mark = false;
            while (num > 0){
                mod = num%10;
                if (mod == 3 || mod == 4 || mod == 7) {
                    mark = false;
                    break;
                }else if(mod == 2|| mod == 5|| mod == 6 || mod == 9){
                    mark = true;
                }
                num /= 10;
            }
            if(mark)count++;
            N--;
        }
        return count;
    }

    /**
     * 国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串，
     *  比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。
     *
     *  为了方便，所有26个英文字母对应摩尔斯密码表如下：
     *  [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---",
     *  "-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-",
     *  "...-",".--","-..-","-.--","--.."]
     *  给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。
     *  例如，"cab" 可以写成 "-.-..--..."，(即 "-.-." + "-..." + ".-"字符串的结合)。
     *  我们将这样一个连接过程称作单词翻译。
     * @param words
     * @return
     */
    public int uniqueMorseRepresentations(String[] words){
        if(words == null) return 0;
        HashSet<String> list = new HashSet<String>();
        String[] str = new String[]{
                ".-","-...","-.-.","-..",".","..-.","--.","....","..",
                ".---","-.-",".-..","--","-.","---",".--.","--.-",".-.",
                "...","-","..-", "...-",".--","-..-","-.--","--.."};
        for(String s: words){
            StringBuilder res = new StringBuilder();
            for(int i = 0; i < s.length(); i++){
                res.append(str[s.charAt(i) - 'a']);
            }
            list.add(res.toString());

        }
        return list.size();
    }

    /**
     * 给定一个段落（paragraph）和一个禁用单词列表（banned）。返回出现次数最多，同时不在禁用列表中的单词。
     * 题目保证至少有一个词不在禁用列表中，而且答案唯一。
     *
     * 禁用列表中的单词用小写字母表示，不含标点符号。段落中的单词不区分大小写。答案都是小写字母。
     * @param paragraph
     * @param banned
     * @return
     */
    public String mostCommonWord(String paragraph,String[] banned){
        HashMap<String, Integer> map = new HashMap<String, Integer>();
//        Pattern p = Pattern.compile("[.;,\"\\?!:']");
//        Matcher m = p.matcher(paragraph);
//        paragraph = m.replaceAll("");
//        HashSet<String> set = new HashSet<>();
        List<String> list = Arrays.asList(banned);
        String[] str = paragraph.toLowerCase().split("[^a-z]+");
        int len = str.length, lenb = banned.length;
//        while(--lenb >= 0)set.add(banned[lenb]);
        int i = 0,num = 0;
        String res = "";
        while(i < len){
            str[i] = str[i].toLowerCase();
            if(!list.contains(str[i]))
                map.put(str[i],map.getOrDefault(str[i],0)+1);
            if(map.getOrDefault(str[i],0) >= num){
                res = str[i];
                num = map.getOrDefault(str[i],0);
            }
            i++;
        }
        return res;
    }

    /**
     * 给定一个由空格分割单词的句子S 。每个单词只包含大写或小写字母。
     * 我们要将句子转换为“Goat Latin”
     * 山羊拉丁文的规则如下：

     如果单词以元音开头（a, e, i, o, u），在单词后添加"ma"。
     例如，单词"apple"变为"applema"。

     如果单词以辅音字母开头（即非元音字母），移除第一个字符并将它放到末尾，之后再添加"ma"。
     例如，单词"goat"变为"oatgma"。

     根据单词在句子中的索引，在单词最后添加与索引相同数量的字母'a'，索引从1开始。
     例如，在第一个单词后添加"a"，在第二个单词后添加"aa"，以此类推。

     返回将 S 转换为山羊拉丁文后的句子
     * @param S
     * @return
     */
    public String toGoatLatin(String S){
        String[] str = S.split(" ");
        StringBuilder res = new StringBuilder();
        int len = str.length;
        for(int i = 0; i < len; i++){
            int num = i+1;
            if(isVowelStrat(str[i])){
                res.append(str[i]).append("ma");
            }else{
                res.append(str[i].substring(1)).append(str[i].charAt(0)).append("ma");
            }
            while(num-- >0)
                res.append("a");
            if(i < len-1)
                res.append(" ");
        }
        return res.toString();
    }
    public boolean isVowelStrat(String str){
        char ch = str.charAt(0);
        return ch == 'a' || ch == 'i' || ch == 'o'|| ch == 'u' ||  ch == 'e' ||
                ch == 'A' || ch == 'I' || ch == 'O'|| ch == 'U' ||  ch == 'E';
    }

    /**
     * 判断两个字符串，其中之一交换两个字符是否和两一个字符串相等。相等返回true,不等返回false
     * @param A
     * @param B
     * @return
     */
    public boolean buddyStrings(String A, String B){
        int lena = A.length();
        if(lena != B.length())return false;
        if(A.equals(B)){
            for(int i = 0; i < 26; i++)
            {
                char ch = (char)('a'+i);
                if(A.indexOf(ch) != A.lastIndexOf(ch))
                    return true;
            }
        }else {
            char[] charsA = A.toCharArray();
            char[] charsB = B.toCharArray();
            int[] indexDifferent = new int[2];
            int k = 0;
            for (int i = 0; i < lena; i++) {
                if (charsA[i] != charsB[i]) {
                    try {
                        indexDifferent[k++] = i;
                    } catch (IndexOutOfBoundsException e) {
                        return false;
                    }
                }
            }
            return k == 2 && charsA[indexDifferent[0]] == charsB[indexDifferent[1]] && charsA[indexDifferent[1]] == charsB[indexDifferent[0]];
        }
        return false;
    }

    /**
     * 你将得到一个字符数组A。
     * 如果经过任意次数的移动 S == T，那么两个字符串 S和T是特殊等价的。
     * 一次移动包括选择两个索引i和j，且i%2 == j%2,交换s[i] 和 s[j]。
     * 现在规定，A 中的特殊等价字符串组是 A 的非空子集 S，这样不在 S 中的任何字符串
     * 与 S 中的任何字符串都不是特殊等价的。
     * 返回 A 中特殊等价字符串组的数量。
     */
    // 没做出来
    public int numSpecialEquivGroups(String[] A){
        HashSet<String> set = new HashSet<>();
        for(int i = 0; i < A.length; i++){
            set.add(A[i]);
        }
        return set.size();
    }

    /**
     * 寻找两个字符串的最大公约数字符串，
     * 如果存在就返回这个公约数字符串，如果不存在就返回空串。
     * @param str1
     * @param str2
     * @return
     */
    public String gcdOfStrings(String str1, String str2){
        if(str1 == null || str2 == null)return "";
        String longStr = str1.length() > str2.length()? str1:str2;
        String shortStr = str1.length() > str2.length()? str2:str1;
        while(longStr.length() != shortStr.length()){
            if(longStr.indexOf(shortStr) == 0){
                String temp = longStr.substring(shortStr.length(),longStr.length());
                longStr = temp.length() > shortStr.length()? temp:shortStr;
                shortStr = temp.length() > shortStr.length()? shortStr:temp;
            }else{
                return "";
            }
        }
        if(longStr.equals(shortStr))return longStr;
        return "";
    }

    /**
     * 你有一个日志数组 logs。每条日志都是以空格分隔的字串。
     * 对于每条日志，其第一个字为字母数字标识符。然后，要么：
     * 标识符后面的每个字将仅由小写字母组成，或；
     * 标识符后面的每个字将仅由数字组成。
     * 我们将这两种日志分别称为字母日志和数字日志。保证每个日志在其标识符后面至少有一个字。
     * 将日志重新排序，使得所有字母日志都排在数字日志之前。字母日志按内容字母顺序排序，忽略标识符；
     * 在内容相同时，按标识符排序。数字日志应该按原来的顺序排列。
     * 返回日志的最终顺序。
     * @param logs
     *
     *  (1) 先区分内容日志和数字日志
     *  (2) 分别对内容日志和数字日志排序。
     * @return
     */
    public String[] reorderLogFiles(String[] logs){
        List<String> letterList = new ArrayList<>();
        List<String> digitList = new ArrayList<>();
        for(String str: logs){
            char ch = str.charAt(str.indexOf(' ')+1);
            if(ch >= '0' && ch <= '9'){
                digitList.add(str);
            }else letterList.add(str);
        }
        letterList.sort(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                int index1 = o1.indexOf(' ') + 1;
                int index2 = o2.indexOf(' ') + 1;
                String temp1 = o1.substring(index1);
                String temp2 = o2.substring(index2);
                int t = temp1.compareTo(temp2);
                if(t == 0)return o1.substring(0, index1).compareTo(o2.substring(0,index2));
                return t;
            }
        });
        letterList.addAll(digitList);
        return letterList.toArray(logs);
    }

    public String[] reorderLogFiles1(String[] logs){
        Arrays.sort(logs,(logs1,logs2) ->{
            String[] split1 = logs1.split(" ", 2);
            String[] split2 = logs2.split(" ", 2);
            boolean isDigit1 = Character.isDigit(split1[1].charAt(0));
            boolean isDigit2 = Character.isDigit(split2[1].charAt(0));
            if(!isDigit1 && !isDigit2){
                int cmp = split1[1].compareTo(split2[1]);
                if(cmp == 0)return split1[0].compareTo(split2[0]);
                return cmp;
            }
            return isDigit1?(isDigit2?0:1):-1;
        });
        return logs;
    }

    /**
     * 寻找邮件数组中的唯一邮件数量
     * @param emails
     * @return
     */
    public int numUniqueEmails(String[] emails){
        Set<String> set = new HashSet<>();
        int i = 0;
        while(i < emails.length){
            StringBuilder res = new StringBuilder();
            int index = emails[i].indexOf('@');
            int j = 0;
            while(j < index){
                if(emails[i].charAt(j) == '.')
                {
                    j++;
                    continue;
                }
                if(emails[i].charAt(j) == '+'){
                    break;
                }
                res.append(emails[i].charAt(j));
                j++;
            }
            res.append(emails[i].substring(index));
            set.add(res.toString());
            i++;
        }
        return set.size();
    }

    /**
     * 你的朋友正在使用键盘输入他的名字 name 偶尔，在键入字符 c的时候，按键可能会被长按，
     * 而字符可能被输入1次或多次。
     * 你将会检查键盘输入的字符 typed 。如果它对应的可能是你朋友的名字（其中一些字符可能被长按），
     * 那么就返回True.
     * @param name
     * @param typed
     * @return
     */
    public boolean isLongPressedName(String name, String typed){
        int i = 0, j = 0;
        while(j < typed.length())
        {
            if(i < name.length() && name.charAt(i) == typed.charAt(j)){
                i++;
                j++;
            }else if(i > 0 && name.charAt(i-1) == typed.charAt(j))
            {
                    j++;
            }else break;
        }
        return i==name.length();
    }

    /**
     * 给定一个字符串S,返回“反转后”字符串，
     * 其中不是字母的字符都保留在原地，
     * 而所有字母的位置发生反转。
     * @param S
     * @return
     */
    public String reverseOnlyLetters(String S){
        char[] chars = S.toCharArray();
        int i= 0, len = S.length()-1;
        while(i < len){
            if(isLetter(chars[i]) && isLetter(chars[len])){
                char temp = chars[i];
                chars[i] = chars[len];
                chars[len] = temp;
            }else if(!isLetter(chars[i]) && isLetter(chars[len])){
                i++;
                continue;
            }else if(isLetter(chars[i]) && !isLetter(chars[len])){
                len--;
                continue;
            }
            i++;
            len--;
        }
        return new String(chars);
    }
    public boolean isLetter(char ch){
        return (ch >='a' && ch <= 'z') || (ch >='A' && ch <= 'Z');
    }
    public static void main(String args[]){
        String s = "krmyfshbspcgtesxnnljhfursyissjnsocgdhgfxubewllxzqhpasguvlrxtkgatzfybprfmmfithphckksnvjkcvnsqgsgosfxc";
        String haystack = "hello", needle="ll";
        String s1 = "cjcjhoxgolccskkhxjzhhkdawonihhkffmdqvvsaehdzvrpkjlygflojqmyrjowcoeskgmjmzvnnbttmddngdptgfestestuwhmqnonntgsrryqwrrsmoaigubcqpeeuzogjdzpevtyrdpatoesrybpdyanheojxdvrdwmsxuidsmdpvfgbirimlkpykrcfhqdpmabgyuzbretlxlogyrkaibxuddotsrkcjryznypmesxjvbjfmrntggyjooackhsnycjqwlwytvdqmmdbohvjnmljyfxrpmxsfugnzrftvxgckobepplrgwaoubvwhzjzfzuvdgrnjrtvtigbwgjqldfzibfwhqqumqzioxuuwdywgnrwcgvisnaobzvdtyqwijjirefqkpodnaitnfrfjlcctxatvtwstgyzfpkwyuwnxobiwsmetbcfucpqkpdgnipynqgnzeclgqnzaqbjqnnnmoktvjtbawdhfnyuvrfdfofookjtgaldcjxrznixtarlrowbiuwillpiecoyyijgwxidtsjnvwyspdlbtfrlmddklhcvgaoakqnmxybotqlvtmazymyfedkwtmescpdtzqhqjmhawefbehzmbeypbhagiceriokxkfubroofgwmscjtojwuzmmzmzcgpaynrrspavdvyykzohcgpkoplbiwrjxjsohutswhmiobjpjhuavtuxllubqyqkgeqvqvjplqojqlsypiygjwwwfzywmorvqcnuzltwhrmvwlpcqfsluyjmuefqopoopwwtdwuizynteboqgxubilmuzgrbxbiwrslgnsetrskubqptmocnfktkdbfotubwczlgcvujihnawvfupwbunowcjxhndpvmehithtqkmaurpshxgkrqlzboskouryhawbuwowaorblnvodyvcayhleqsdsaegpaxelhejopjlxrhrxulfeamzvzlgpqbladorcxbvajixteorxjymrkrmaalpvivcstxmhwjxpnihyudcgybnlmbtgcoqcsvlsqsdmxgmohoauswkmbwftjwpvddoewwmeafoyluxwwdgykcwoodenlbuhbwhdvllydqhncvffhpbwixefvekchkohatajqldjpbfxcnuwhghvcunzpovplpxylxsesrnunrpudnqehzmwmorwuhbzhmjrsykhhhxxmvrtsofgvpdaaiwuelzkvfwbkzfjtsgnsdjdlduvwptiuhxhsrqboywphdadubfkjrkyedlcdopwatwbbwpbqldqedwgfwhxzxwbwftqeelfwhbtzndkgoiuqchuqqtvrdzwjzwsjkblnvznqetprqojavyeafeclffjjdjxdjthebxodasuxfjxheupavtnhhjxilmvdmbjboyxujvdiawkqhfjswgszbzhgjxawwtsxfrzoqzxgkrcrljvliphwfomninzabzfeetlvtbijtlnruglkqvyqeupcbqanvhficivbbpuirvsxiouepqwrupbkggzknrnckzgiivdioyprxazcjlslgfvaeyrnswvmvjgouusvnkytylsunaywlxibnhhmtpkxvvausdfoxdkzjnzesoqvawbezvlpswjfyblnuiywvnqwtsbwzokppwhyjxmhijhnlyyomsphaxpncuficqulwafoanyxzxhflikydeikbtjszbcuhcjpchfzkszcfdjrxyunlqndubmwftmfupfmnwhycogvulbogfpgsniqlcpqpwdnnzzxyjqzotrzrvioetebormazaujndoqurbthzydurdzycfnggebhrnxcazdrcakfxtqfwditgqeiszllfaxgyrktjdxrmqvhegiufhrmrfoovrihbnpyymxbkdwmqwnpmiiwvvvvvgylwntoahklxkkmprejvqyotkiorqngxrrroizvlbdfxeiyjrnsgajgzmehmwxgzowoannuhkvtkpbzjkltldkxobouokykaxpwqbepzrwdszoxkyudzqflzoatphcehwoqasaoatremxbbrflovpnjcaydxswevvlltwdwpqymtqyirycsgdfjvkrzrjosmecnntkaacryyfljtzryjjbtrtnquzrzkjybdkjprjsgomjgdztkuuomlyxboqirjfsamtxhdkrdsbngacfzdzbyjjssnzfrvdoehpfrwrddybjnfxjhwscildydobhnigwuyypwinbaplwpcukfiejkvydcipedrbsduuttcbiwccqlkbiadzthzbblcmnvdbojukazjpkepatebkqdwssgozgylkgucqhwpjhetvozzzztrhokejmhbmmnpfjhcxffhogdznlpyvjeoiyqtboaixkpooksqfsmjvtbzigfqfktsqlbwmcvakerpkwscaetamvbdpzunghpsczzxvhwcvkvwnqebhypggjeoyjeaunxztdukmzluhvtreqgegwsopijodmogzbdsnbprlegomcqgpecywxqmedufnthcgcnkzyyyedpjgjycvvyqmnfvzpfzztbdbfnirgycgrbbkxttmqdbdmqhsoznoxdkrnwgmrwzduxhitmaeembgigvijadwmcpqrlbnfjlnjjgfhqwwlonslonidyozoupnfxshtzhvfyplepbppbtpownwhksvbxibfwyljoxcsogkyeuuyqbwdlvlcegqpdgnovlfhwdxmbhpyrqzdlniykfgrpulgswdweaszaqdjdsvtayjrmgfjytljkuhwekcgvouzrtnnbtyvtmgozkrciyihwkjkidqmzintagbirfzvujjlhtqzzqzjbacfgssbuemwdlndmhpaeiwykbocsjdjdcvviqgkdqatsjgiroxlrouuqblworskvakejfnzonwnrjhdumwvyazmdtdpcinezklxqctoqfzisfqjobyskasadoznotxdlrewciajzjrsbbreutwlvluxbttwmfcvnpuvlbxxgimyrqgmwvwiebntynlfqdshrazrruihngjdxvxbqsblyjilxwfplmxftwcjpikyffnhkjkhszkyfnynfzdadpubzuonqiczsquqxyryjuemhvnazjfrhlcdnuyjwzaqlxtcjngolsdlomxpjwzcxihrrqcexhznulqylzbypgtuodhbaiifcweqoqaszhwaovopmtqtokkfhzcqjaxozfqokefvauthvvarttyklvfckqtznkntnlheixrqzhrxysyspoonasjceyuptherlmztfpwuqdgnnthteatndktklupnyopbojwtblgbigijmcmqplduxgkqafsfxphgcsbusrvhlhbnjnlwtlkemzoaioynfckphiigkocwtvvnjenmmnqcmapdcrrjohjqbslsxsjfmpgsofgaiavagjkntbrqxdkaruvpflfsddmrdwkqvmclcutbzprmxrniudowjbkmwxtxxhvlqrftdfzlowcfjzgkjtaxfwvmsditltkkzvyyqnqumimzjnwrozpugdfiouphxxefpsmsletfnqlbwcuokvnkoruzqeetslhocgyvuoqtruepmdcyfwcbdgagceqbmwzhbvltfoptrjiftboptppeoelsodqeoklwbtizuusuhfbhkwnzcbfhsaiytaxileiygjboclgbubnaxdlkclyfccorjhpdbtyjlvvgxhcuafvvnqdfnlnqimukpanmfcrglbhlkrvvpnqdqvfuxqtzryrvbreotcixznxncysjkmjjdcikvovjfdshonxvbgqagaaibvtzqrbrqqkvlmsuvcsxhibmtvlprekjapyomcfkhqsgnthplgfpjlgleurzazpyypcajnwcwyvtdlgxndzahpmpvdohisglixqaxpsybkvticitmxsndxbypgdbejxjyrohdpkxxekavlihqedevvnyftsltkzgzkupzdfffxyrknzteimreoylguoloummupgtaxqptvgxwniqkwzkposbkjogogkdrikpklhfbuwapoydbnquwcjtjlditwhuaxnjpixcbnwxedcxwpacbckymdruriuxwnfaqrtyjinveaczznfqlcsafppbqpmkirmeogvjshbsrjikoabvujavsxteydxxoegbfoiybtnznnoilqhbjevvpnmvftxclqsqmilxjliniwhlzpghcflaqbxgsvyyczwwtlmraslwywjwwudemhqgmvymwvikxgxaxfcjpeyuigarjorlubvkmzdxiqnxofknnvamtavtrdglkcyohpfjcqdeiglfmbnrnsnnqbylxucxoghktmaermomigkcdehauitjxbjynjmmedrbarvnpoynrsgvzgzodrqudqfygfhxmxfaciritlxwgxthauxenxzmwehrerfgbdgikfglqjqbmsvugzpvfsiwvhinhzqqpbowopgzsogyetpfloxvskmbpqsfsaubfedghltkxlqvztkeacabnufzfvpncywdxgauxftpsrbfxaqiofcoapxnjkyhtffanywxaqhjeoocffxqewtslmepmburkymlhdkcwlzulctqvkphseycvoecovmkezhzyegbnxyyqiovrmvcqfoezbsjzfanhtmolelipkepnkwtuhmhlxdegsbldonsicehibbhinqizgbbnnwmxbuffkvnfbniypqnimfwvnzyifgymmelukwkfmcyhnejjeycesgsclhcnbhejiwjpcwulqqnsnrgalkoidnrtrhppjobatdsmviplaveockpsgbsxhogohfemwuobjmduhntxpfpikfnyxlkfjnupmneuxlhewkdyqgquidieyggmragzltofpmrcyuvfsvzdroymqyyxffmofndwxeviesgadmewjnnllpvijhnnxpharnsxuackebkfmqldfhfobnqnduztsfdzxxhezhmisdlnppnldsimhzehxxzdfstzudnqnbofhfdlqmfkbekcauxsnrahpxnnhjivpllnnjwemdagseivexwdnfomffxyyqmyordzvsfvuycrmpfotlzgarmggyeidiuqgqydkwehlxuenmpunjfklxynfkipfpxtnhudmjbouwmefhogohxsbgspkcoevalpivmsdtabojpphrtrndioklagrnsnqqluwcpjwijehbnchlcsgsecyejjenhycmfkwkulemmygfiyznvwfminqpyinbfnvkffubxmwnnbbgziqnihbbihecisnodlbsgedxlhmhutwknpekpilelomthnafzjsbzeofqcvmrvoiqyyxnbgeyzhzekmvoceovcyeshpkvqtcluzlwckdhlmykrubmpemlstweqxffcooejhqaxwynaffthykjnxpaocfoiqaxfbrsptfxuagxdwycnpvfzfunbacaektzvqlxktlhgdefbuasfsqpbmksvxolfpteygoszgpowobpqqzhnihvwisfvpzguvsmbqjqlgfkigdbgfrerhewmzxnexuahtxgwxltiricafxmxhfgyfqduqrdozgzvgsrnyopnvrabrdemmjnyjbxjtiuahedckgimomreamtkhgoxcuxlybqnnsnrnbmflgiedqcjfphoycklgdrtvatmavnnkfoxnqixdzmkvbulrojragiuyepjcfxaxgxkivwmyvmgqhmeduwwjwywlsarmltwwzcyyvsgxbqalfchgpzlhwiniljxlimqsqlcxtfvmnpvvejbhqlionnzntbyiofbgeoxxdyetxsvajuvbaokijrsbhsjvgoemrikmpqbppfasclqfnzzcaevnijytrqafnwxuirurdmykcbcapwxcdexwnbcxipjnxauhwtidljtjcwuqnbdyopawubfhlkpkirdkgogojkbsopkzwkqinwxgvtpqxatgpummuolouglyoermietznkryxfffdzpukzgzktlstfynvvedeqhilvakexxkpdhoryjxjebdgpybxdnsxmticitvkbyspxaqxilgsihodvpmphazdnxgldtvywcwnjacpyypzazruelgljpfglphtngsqhkfcmoypajkerplvtmbihxscvusmlvkqqrbrqztvbiaagaqgbvxnohsdfjvovkicdjjmkjsycnxnzxictoerbvryrztqxufvqdqnpvvrklhblgrcfmnapkumiqnlnfdqnvvfauchxgvvljytbdphjroccfylckldxanbubglcobjgyielixatyiashfbcznwkhbfhusuuzitbwlkoeqdosleoepptpobtfijrtpoftlvbhzwmbqecgagdbcwfycdmpeurtqouvygcohlsteeqzuroknvkoucwblqnftelsmspfexxhpuoifdgupzorwnjzmimuqnqyyvzkktltidsmvwfxatjkgzjfcwolzfdtfrqlvhxxtxwmkbjwoduinrxmrpzbtuclcmvqkwdrmddsflfpvurakdxqrbtnkjgavaiagfosgpmfjsxslsbqjhojrrcdpamcqnmmnejnvvtwcokgiihpkcfnyoiaozmekltwlnjnbhlhvrsubscghpxfsfaqkgxudlpqmcmjigibglbtwjobpoynpulktkdntaethtnngdquwpftzmlrehtpuyecjsanoopsysyxrhzqrxiehlntnknztqkcfvlkyttravvhtuavfekoqfzoxajqczhfkkotqtmpovoawhzsaqoqewcfiiabhdoutgpybzlyqlunzhxecqrrhixczwjpxmoldslognjctxlqazwjyundclhrfjzanvhmeujyryxquqszciqnouzbupdadzfnynfykzshkjkhnffykipjcwtfxmlpfwxlijylbsqbxvxdjgnhiurrzarhsdqflnytnbeiwvwmgqrymigxxblvupnvcfmwttbxulvlwtuerbbsrjzjaicwerldxtonzodasaksybojqfsizfqotcqxlkzenicpdtdmzayvwmudhjrnwnoznfjekavksrowlbquuorlxorigjstaqdkgqivvcdjdjscobkywieaphmdnldwmeubssgfcabjzqzzqthljjuvzfribgatnizmqdikjkwhiyicrkzogmtvytbnntrzuovgckewhukjltyjfgmrjyatvsdjdqazsaewdwsgluprgfkyinldzqryphbmxdwhflvongdpqgeclvldwbqyuueykgoscxojlywfbixbvskhwnwoptbppbpelpyfvhzthsxfnpuozoydinolsnolwwqhfgjjnljfnblrqpcmwdajivgigbmeeamtihxudzwrmgwnrkdxonzoshqmdbdqmttxkbbrgcygrinfbdbtzzfpzvfnmqyvvcyjgjpdeyyyzkncgchtnfudemqxwycepgqcmogelrpbnsdbzgomdojiposwgegqertvhulzmkudtzxnuaejyoejggpyhbeqnwvkvcwhvxzzcsphgnuzpdbvmateacswkprekavcmwblqstkfqfgizbtvjmsfqskoopkxiaobtqyioejvyplnzdgohffxchjfpnmmbhmjekohrtzzzzovtehjpwhqcugklygzogsswdqkbetapekpjzakujobdvnmclbbzhtzdaibklqccwibcttuudsbrdepicdyvkjeifkucpwlpabniwpyyuwginhbodydlicswhjxfnjbyddrwrfpheodvrfznssjjybzdzfcagnbsdrkdhxtmasfjriqobxylmouuktzdgjmogsjrpjkdbyjkzrzuqntrtbjjyrztjlfyyrcaaktnncemsojrzrkvjfdgscyriyqtmyqpwdwtllvvewsxdyacjnpvolfrbbxmertaoasaqowhechptaozlfqzduykxozsdwrzpebqwpxakykouoboxkdltlkjzbpktvkhunnaowozgxwmhemzgjagsnrjyiexfdblvziorrrxgnqroiktoyqvjerpmkkxlkhaotnwlygvvvvvwiimpnwqmwdkbxmyypnbhirvoofrmrhfuigehvqmrxdjtkrygxafllzsieqgtidwfqtxfkacrdzacxnrhbeggnfcyzdrudyzhtbruqodnjuazamrobeteoivrzrtozqjyxzznndwpqpclqinsgpfgobluvgocyhwnmfpufmtfwmbudnqlnuyxrjdfczskzfhcpjchucbzsjtbkiedykilfhxzxynaofawluqcifucnpxahpsmoyylnhjihmxjyhwppkozwbstwqnvwyiunlbyfjwsplvzebwavqoseznjzkdxofdsuavvxkptmhhnbixlwyanuslytyknvsuuogjvmvwsnryeavfglsljczaxrpyoidviigzkcnrnkzggkbpurwqpeuoixsvriupbbvicifhvnaqbcpueqyvqklgurnltjibtvlteefzbazninmofwhpilvjlrcrkgxzqozrfxstwwaxjghzbzsgwsjfhqkwaidvjuxyobjbmdvmlixjhhntvapuehxjfxusadoxbehtjdxjdjjfflcefaeyvajoqrpteqnzvnlbkjswzjwzdrvtqquhcquiogkdnztbhwfleeqtfwbwxzxhwfgwdeqdlqbpwbbwtawpodcldeykrjkfbudadhpwyobqrshxhuitpwvudldjdsngstjfzkbwfvkzleuwiaadpvgfostrvmxxhhhkysrjmhzbhuwromwmzheqnduprnunrsesxlyxplpopznucvhghwuncxfbpjdlqjatahokhckevfexiwbphffvcnhqdyllvdhwbhublnedoowckygdwwxulyofaemwweoddvpwjtfwbmkwsuaohomgxmdsqslvscqocgtbmlnbygcduyhinpxjwhmxtscvivplaamrkrmyjxroetxijavbxcrodalbqpglzvzmaefluxrhrxljpojehlexapgeasdsqelhyacvydovnlbroawowubwahyruoksobzlqrkgxhspruamkqthtihemvpdnhxjcwonubwpufvwanhijuvcglzcwbutofbdktkfncomtpqbuksrtesnglsrwibxbrgzumlibuxgqobetnyziuwdtwwpoopoqfeumjyulsfqcplwvmrhwtlzuncqvromwyzfwwwjgyipyslqjoqlpjvqvqegkqyqbullxutvauhjpjboimhwstuhosjxjrwiblpokpgchozkyyvdvapsrrnyapgczmzmmzuwjotjcsmwgfoorbufkxkoirecigahbpyebmzhebfewahmjqhqztdpcsemtwkdefymyzamtvlqtobyxmnqkaoagvchlkddmlrftbldpsywvnjstdixgjiyyoceiplliwuibworlratxinzrxjcdlagtjkoofofdfrvuynfhdwabtjvtkomnnnqjbqaznqglcezngqnypingdpkqpcufcbtemswiboxnwuywkpfzygtswtvtaxtccljfrfntiandopkqferijjiwqytdvzboansivgcwrngwydwuuxoizqmuqqhwfbizfdlqjgwbgitvtrjnrgdvuzfzjzhwvbuoawgrlppebokcgxvtfrzngufsxmprxfyjlmnjvhobdmmqdvtywlwqjcynshkcaoojyggtnrmfjbvjxsempynzyrjckrstodduxbiakrygolxlterbzuygbampdqhfcrkypklmiribgfvpdmsdiuxsmwdrvdxjoehnaydpbyrseotapdrytvepzdjgozueepqcbugiaomsrrwqyrrsgtnnonqmhwutsetsefgtpdgnddmttbnnvzmjmgkseocwojrymqjolfgyljkprvzdheasvvqdmffkhhinowadkhhzjxhkkscclogxohjcjc";
        //System.out.println('1' - '0');
        String str = "`~!@#$%^&*()-_+={[}]|\\:;\"',.>/?·！@#￥%……&*（）——+【】；：’‘“”，《。》、？";
        System.out.println(str.replaceAll("[\\pP\\p{Punct}]", ""));
        String paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.";
        String[] banned = {"hit"};
        String ransomNote = "124";
        String magazine = "132456";
        System.out.println('a' * 5);
        char[] chars = new char[]{'a','a','a','b','b','c','c','c'};
        String[] words = new String[]{"gin", "zen", "gig", "msg"};
        String moves = "LL";
        int c = 'a'; // 65 --- 90
//        System.out.print(c);
        String[] strs = new String[]{"flower","flow","flight"};
        String[]  A = {"a","b","c","a","c","c"};
        String B = "abab";
        String S = "Test1ng-Leet=code-Q!";
        String[] email = {"test.email+alex@leetcode.com", "test.email@leetcode.com"};
        String name = "alex";
        String typed = "aaleex";
        String  mark = new Solution().reverseOnlyLetters(S);
        System.out.print(mark);
//        int sum = new Solution().romanToInt2("MCMXCIV");
//        String sum = new Solution().longestCommonPrefix(strs);
//        System.out.print(sum);
    }
}
