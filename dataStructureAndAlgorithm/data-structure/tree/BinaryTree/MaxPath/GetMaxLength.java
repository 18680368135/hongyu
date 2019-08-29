package tree.BinaryTree.MaxPath;

import java.util.HashMap;

public class GetMaxLength {

    public int getMaxLength(Node head, int sum){

        HashMap<Integer, Integer> sumMap = new HashMap<Integer, Integer>();
        sumMap.put(0, 0);

        return preOrder(head, sum, 0, 1, 0,sumMap);
    }

    public int preOrder(Node head, int sum, int presum, int level, int maxLen, HashMap<Integer, Integer> sumMap){
        if(head == null) return maxLen;

        int cursum = presum + head.value;

        if(!sumMap.containsKey(cursum))
            sumMap.put(cursum, level);
        if(sumMap.containsKey(cursum - sum)){
            maxLen = Math.max(level -sumMap.get(cursum-sum), maxLen);
        }

        maxLen = preOrder(head.left, sum, cursum, level+1, maxLen, sumMap);
        maxLen = preOrder(head.right, sum, cursum, level+1, maxLen, sumMap);

        return maxLen;
    }
}



class Node{
    public Node left;
    public Node right;
    public int value;


    public Node(int data){
        this.value = data;
    }

}
