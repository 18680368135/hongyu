package tree.BinaryTree.RecursiveTraversal;

public class OrderRecur {

    public void preOrderRecur(Node head){
        if(head == null){
            return ;
        }
        System.out.print(head.value + " ");
        preOrderRecur(head.left);
        preOrderRecur(head.right);

    }



    public void inOrderRecur(Node head){
        if(head == null){
            return;
        }

        inOrderRecur(head.left);
        System.out.print(head.value + " ");
        inOrderRecur(head.right);

    }

    public void posOrderRecur(Node head){
        if(head == null){
            return;
        }
        posOrderRecur(head.left);
        posOrderRecur(head.right);
        System.out.print(head.value);

    }



}


class Node{
    public Node right;
    public Node left;
    public int value;


    public Node(int data){
        this.value = data;
    }
}
