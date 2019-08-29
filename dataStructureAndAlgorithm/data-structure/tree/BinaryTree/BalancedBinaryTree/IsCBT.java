package tree.BinaryTree.BalancedBinaryTree;

import java.util.LinkedList;
import java.util.Queue;

public class IsCBT {

    public boolean isCBT(Node head){
        if(head == null) return true;

        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(head);

        Node l = null;
        Node r = null;

        boolean leaf = false;

        while(queue.isEmpty()){
            head = queue.poll();
            l = head.left;
            r = head.right;

            if((leaf && ( l!=null || r != null)) || (l == null && r != null)){
                return false;
            }

            if(l != null){
                queue.offer(l);
            }

            if(r != null){
                queue.offer(r);
            }else{
                leaf = true;
            }
        }
        return true;
    }
}
