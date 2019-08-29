package tree.BinaryTree.BalancedBinaryTree;

public class IsBalancedBinaryTree {

    public ReturnType process(Node head){
        if(head == null) return new ReturnType(true, 0);

        ReturnType leftData = process(head.left);
        ReturnType rightData = process(head.right);

        int height = Math.max(leftData.height, rightData.height) + 1;
        boolean isBalanced = leftData.isBalanced && rightData.isBalanced && Math.abs(leftData.height-rightData.height) < 2;

        return new ReturnType(isBalanced,height);
    }

    public boolean isBalanced(Node head){
        return process(head).isBalanced;
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


