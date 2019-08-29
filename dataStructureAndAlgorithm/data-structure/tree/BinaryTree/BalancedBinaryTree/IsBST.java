package tree.BinaryTree.BalancedBinaryTree;

public class IsBST {


    //morris先序
    public void preMorris(Node head){
        if(head == null) return;

        Node cur = head;
        Node mostRight = null;

        while(cur != null){
            mostRight = head.left;
            if(mostRight != null){
                while (mostRight.right != null && mostRight.right != cur){
                    mostRight = mostRight.right;
                }
                if(mostRight.right == null){
                    mostRight.right = cur;
                    System.out.print(cur.value + " ");
                    cur = cur.left;

                    continue;

                }else {

                    mostRight.right = null;
                }
            }else {

                System.out.println(cur.value + " ");
            }
            cur = cur.right;
        }
        System.out.println();
    }

    //morris中序

    public void morrisIn(Node head){
        if(head == null) return;

        Node cur = head;
        Node mostRight = null;

        while(cur != null){
            mostRight = head.left;
            if(mostRight != null){
                while(mostRight.right != null && mostRight.right != cur){
                    mostRight = mostRight.right;
                }

                if(mostRight.right == null){
                    mostRight.right = cur;
                    cur = cur.left;

                    continue;

                }else {

                    mostRight.right = null;
                }
            }
            System.out.print(cur.value + " ");
            cur = cur.right;
        }
        System.out.println();
    }

    //morris后序
    public void posMorris(Node head){
        if(head == null) return;

        Node cur = head;
        Node mostRight = null;

        while(cur != null){
            mostRight = head.left;
            if(mostRight != null){
                while(mostRight.right != null && mostRight.right != cur){

                    mostRight = mostRight.right;
                }
                if(mostRight.right == null){
                    mostRight.right = cur;
                    cur = cur.left;
                    continue;
                }else{

                    mostRight.right = null;
                }
            }

            cur = cur.right;
        }

    }

    public static void printEdge(Node head){

        Node tail = reverseEdge(head);
        Node cur = tail;
        while(cur != null){
            System.out.print(cur.value + " ");
            cur = cur.right;
        }

    }

    public static Node reverseEdge(Node from){
        Node pre = null;
        Node next = null;
        while(from != null){
            next = from.right;
            from.right = pre;
            pre = from;
            from  = next;
        }

        return pre;

    }


    public boolean isBST(Node head){
        boolean res = true;
        if(head == null) return res;

        Node pre = null;
        Node cur = head;
        Node mostRight = null;

        while(cur != null){

            mostRight = cur.left;
            if(mostRight != null){
                while(mostRight.right != null && mostRight.right != cur){
                    mostRight = mostRight.right;
                }
                if(mostRight.right == null){
                    mostRight.right = cur;
                    cur = cur.left;
                    continue;
                }else {
                    mostRight.right = null;
                }
                if(pre != null && pre.value > cur.value){
                    res = false;
                }
            }
            pre = cur;
            cur = cur.right;
        }
        return res;
    }
}
