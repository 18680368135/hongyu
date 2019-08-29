package tree.BinaryTree.NoRecursiveTraversal;

import java.util.Stack;

public class OrderRecur {


    public void preOrderRecur(Node head){
        if(head != null) {
            System.out.println("preOrder is :");
            Stack<Node> stack = new Stack<Node>();
            stack.add(head);
            if (!stack.isEmpty()) {
                head = stack.pop();
                System.out.print(head.value + " ");
                if (head.right != null) {
                    stack.push(head.right);
                }

                if (head.left != null) {
                    stack.push(head.left);
                }
            }
        }
    }

    public void inOrderRecur(Node head){
        if(head != null) {
            System.out.println("InOrderRecur is :");
            Stack<Node> stack = new Stack<Node>();
            while (head != null || !stack.isEmpty()) {
                if(head != null){
                    stack.push(head);
                    head = head.left;
                } else
                    {
                    head = stack.pop();
                    System.out.print(head.value + " ");
                    head = head.right;
                    }
            }
        }


    }


    public void posOrderRecur(Node head){
        if(head != null){
            System.out.println("posOrderRecur is :");
            Stack<Node> stack = new Stack<Node>();

            stack.push(head);
            Node cur = null;

            while(! stack.isEmpty()){
                cur = stack.peek();

                if(cur.left!= null && head != cur.left && head != cur.right){
                    stack.push(cur.left);
                }else if(cur.right != null && head != cur.right)
                {
                    stack.push(cur.right);
                }
                else{
                    head = stack.pop();
                    System.out.print(head.value + " ");
                    head = cur;
                }
            }
        }

        System.out.println();
    }

    public void posOrderRecur1(Node head){

        if(head != null){
            Stack<Node> s1 = new Stack<Node>();
            Stack<Node> s2 = new Stack<Node>();
            s1.push(head);
            while (!s1.isEmpty()){
                head = s1.pop();
                s2.push(head);
                if(head.left != null){
                    s1.push(head.left);
                }
                if(head.right != null){
                    s1.push(head.right);
                }

            }

            while(!s2.isEmpty()){
                System.out.print(s2.pop().value + " ");
            }

        }
        System.out.println();

    }
}
