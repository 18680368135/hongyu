package tree.BinaryTree.PrintBinaryTreeEdge;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

public class PrintByLevel {

    public void printByLevel(Node head){
        if(head == null) return;

        Queue<Node> queue = new LinkedList<>();

        queue.offer(head);
        Node last = head;
        Node nLast = null;
        int level = 1;

        System.out.print("level "+ (level++) + " : ");
        while(!queue.isEmpty()){
            head = queue.poll();
            System.out.print(head.value + " ");
            if(head.left !=  null){

                queue.offer(head.left);
                nLast = head.left;
            }
            if(head.right != null){
                queue.offer(head.right);
                nLast = head.right;
            }
            /**
             * 队列可以不用空，但是必须判断是不是当了当前层的末尾，
             * 如当前层末尾已到就让last指向当前节点
             */
            if(head == last && !queue.isEmpty()){
                last = nLast;
                System.out.print("\nlevel " + (level++) + ":");
            }
        }
        System.out.println();

    }

    public void printByZigZag(Node head){
        if(head == null) return;
        Deque<Node> dq = new LinkedList<Node>();
        dq.addFirst(head);

        boolean lr = true; //用于判断进入队列的位置是从头部还是尾部。
        int level = 1;
        Node last = head;
        Node nLast = null;
        printLevelAndorientation(level++, lr);

        while(!dq.isEmpty()){
            if(lr){
                head = dq.pollFirst();
                if(head.left != null){
                    dq.offerFirst(head.left);
                    nLast = nLast == null ? head.left : nLast;
                }

                if(head.right != null){
                    dq.offerFirst(head.right);
                    nLast = nLast == null ? head.right : nLast;
                }


            }else{
                head = dq.pollLast();
                if(head.left != null){
                    dq.offerFirst(head.left);
                    nLast = nLast == null ? head.left : nLast;
                }
                if(head.right != null){
                    dq.offerFirst(head.right);
                    nLast = nLast == null ? head.right : nLast;
                }
            }
            System.out.println(head.value + " ");
            if(head == last && !dq.isEmpty()){
                last = nLast;
                lr = !lr;
                nLast = null;  //由于前后交替输出所以每次必须指定nLsat == null

                System.out.println();
                printLevelAndorientation(level, lr);
            }


        }

    }

    public void printLevelAndorientation(int level, boolean lr){
        System.out.print("Level " + (level++) + " from ");
        System.out.print(lr ? "left to right" : "right to left");
    }
}
