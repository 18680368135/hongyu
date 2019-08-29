package tree.BinaryTree.BalancedBinaryTree;

public class PosArrayToBST {


    public Node posArrToBST(int[] posArr){
        if(posArr == null)
            return null;

        return posToBST(posArr, 0, posArr.length-1);
    }

    public Node posToBST(int[] posArr, int start, int end){

        if(start > end) return null;

        Node head = new Node(posArr[end]);
        int less = -1;
        int more = end;
        for(int i = start; i < end; i++){
            if(posArr[end] > posArr[i]){

                less = i;

            }else {

                more = more == end ? i : more;
            }
        }

        head.left = posToBST(posArr, start, less);
        head.right = posToBST(posArr, more, end);

        return head;
    }
}
