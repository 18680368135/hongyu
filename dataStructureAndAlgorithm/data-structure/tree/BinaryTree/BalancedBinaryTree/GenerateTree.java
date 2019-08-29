package tree.BinaryTree.BalancedBinaryTree;

public class GenerateTree {

    public Node generate(int[] sortarr, int start, int end){
        if(start > end){
            return null;
        }
        int mid = (start + end) /2;
        Node head = new Node(sortarr[mid]);

        head.left = generate(sortarr, start, mid-1);
        head.right = generate(sortarr, mid+1, end);
        return head;
    }

    public Node generateTree(int[] sortarr){
        if(sortarr == null) return null;

        return generate(sortarr, 0, sortarr.length-1);
    }
}
