package tree.BinaryTree.FindErrorNode;

import java.util.Stack;

public class FindTwoErrorNode {

    public Node[] getTwoErrorNode(Node head){
        Node[] error = new Node[2];
        Stack<Node> stack = new Stack<Node>();

        Node pre = null;

        while(head != null && !stack.isEmpty()){
            if(head != null){
                stack.push(head);
                head = head.left;
            }else {
                head = stack.pop();

                if(pre != null && pre.value > head.value){
                    error[0] = error[0] == null ? pre : error[0];
                    error[1] = head;
                }
                pre = head;
                head = head.right;
            }
        }
        return error;
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
