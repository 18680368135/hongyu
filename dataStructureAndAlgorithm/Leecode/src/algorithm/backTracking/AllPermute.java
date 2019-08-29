package algorithm.backTracking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AllPermute {
    public List<List<Integer>> permute1(int[] nums){
        List<Integer> numsArray = new ArrayList<Integer>();
        for(int num: nums) numsArray.add(num);
        Collections.sort(numsArray);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        solve1(res,numsArray,0);
        return res;
    }
    public void solve1(List<List<Integer>> res, List<Integer> numsArray, int index ){
        if(index >= numsArray.size()){
            System.out.println("numsarray: "+numsArray);

            List<Integer> nums = new ArrayList<Integer>(numsArray); // 不理解
            res.add(numsArray);
            System.out.println("res: "+res);
        }
        for(int i = index; i < numsArray.size(); i++){

            Collections.swap(numsArray, i, index);
            System.out.println("before: "+numsArray);
            solve1(res, numsArray, index+1);
            System.out.println("after: "+numsArray);
            Collections.swap(numsArray, i, index);
        }
    }
    public List<List<Integer>> permute(int[] nums){
        List<Integer> numsArray = new ArrayList<Integer>();
        for(int num: nums) numsArray.add(num);
        Collections.sort(numsArray);
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> subSet = new ArrayList<Integer>();
        solve(numsArray, used, res, subSet);
        return res;
    }
    public void solve(List<Integer> numsArray, boolean[] used, List<List<Integer>> res, List<Integer> subSet){
        if(subSet.size() == numsArray.size()){
            List<Integer> clone = new ArrayList<Integer>(subSet);
            res.add(clone);
        }
        for(int j = 0; j < numsArray.size(); j++){
            if(used[j] == true) continue;
            subSet.add(numsArray.get(j));
            used[j] = true;
            solve(numsArray, used, res, subSet);
            subSet.remove(numsArray.get(j));   //回退
            used[j] = false;
        }
    }
    public static void main(String args[]){
        int[] nums = new int[]{1,2,3};
        AllPermute per= new AllPermute();
        System.out.println(per.permute(nums));
    }
}
