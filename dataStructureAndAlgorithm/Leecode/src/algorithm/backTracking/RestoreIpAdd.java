package algorithm.backTracking;

import java.util.*;

/**
    IP地址重组还需要再看一下。
 **/
public class RestoreIpAdd {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<String>();
        nextIPnum(s, 0, res, "");
        return res;
    }
    public void nextIPnum(String s,int n,  List<String> res, String str){
        if(n == 4){
            if(s.isEmpty())
                res.add(str);
            return;
        }
        for(int i = 1; i < 4; i++){
            if(s.length() < i) break;
            int val = Integer.parseInt(s.substring(0,i));
            if(val > 255 || String.valueOf(val).length()!= i) continue;
            // 巧妙之处在于把回溯添加在递归函数里面
            nextIPnum(s.substring(i),n+1,res, str+s.substring(0,i)+(n==3?"":"."));
            }
        }


    public static void main(String args[]){
        String str = "25525511135";
        RestoreIpAdd rip = new RestoreIpAdd();
        System.out.println(rip.restoreIpAddresses(str));
    }
}
