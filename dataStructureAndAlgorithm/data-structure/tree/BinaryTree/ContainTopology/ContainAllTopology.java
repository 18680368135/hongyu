package tree.BinaryTree.ContainTopology;

public class ContainAllTopology {

    public static boolean contains(Node t1, Node t2){
        if(t2 == null) return true;
        if(t1 == null) return false;

        return check(t1, t2) || contains(t1.left, t2) || contains(t1.right, t2);
    }

    public static boolean check(Node t1, Node t2){

        if(t2 == null) return true;
        if(t1 == null || t1.value != t2.value) return false;

        return check(t1.left, t2) && check(t1.right, t2);
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