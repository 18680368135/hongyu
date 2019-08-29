package tree.BinaryTree.SerializationAndDeserialization;

import java.util.LinkedList;
import java.util.Queue;

public class PreOrderSeria
{
    /**
     *
     * 通过先序遍历实现二叉树的序列化
     *
     */


    public String seriaPreOreder(Node head){
        if(head == null) return "#!";

        String res = head.value+"!";
        res += seriaPreOreder(head.left);

        res += seriaPreOreder(head.right);
        return res;

    }

    /**
     *
     * 实现先序遍历的二叉树反序列化
     * 队列使用很好，通过出队，不断添加节点
     * 使用递归实现二叉树的重新构建
     *
     */

    public Node reconByPreString(String str){
        String[] values = str.split("!");
        Queue<String> queue = new LinkedList<String>();

        for(int i = 0; i < values.length; i++){
            queue.offer(values[i]);
        }
        return reconPreOrder(queue);
    }

    public Node reconPreOrder(Queue<String> queue){
        String value = queue.poll();
        if(value.equals("#")) return null;
        Node head = new Node(Integer.valueOf(value));
        head.left = reconPreOrder(queue);
        head.right = reconPreOrder(queue);

        return head;
    }


    /**
     * 通过分层实现二叉树的序列化
     *
     *
     * 将每层的非空节点入队，通过判断队列非空不断添加每层节点的值到字符串中
     */
    public String seriaByLevel(Node head){

        if(head == null) return "#!";
        String res = head.value + "!";
        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(head);
        while (!queue.isEmpty()){
            head = queue.poll();

            if(head.left != null){
                res += head.left.value + "!";
                queue.offer(head.left);
            }
            else
                {

                    res += "#!";

                }
            if(head.right != null){

                res += head.right.value + "!";
                queue.offer(head.right);
            }else
                {

                    res += "#!";

                }
        }
        return res;
    }

    /**
     *
     * 通过分层的方式实现二叉树的反序列化
     *
     */
    public Node reconByLevelString(String levelStr){
        String[] value = levelStr.split("!");
        int index = 0;
        Node head = generateNodeByString(value[index++]);
        Queue<Node> queue = new LinkedList<Node>();

        if(head != null) queue.offer(head);

        while(!queue.isEmpty()){
            head = queue.poll();
            head.left = generateNodeByString(value[index++]);
            head.right = generateNodeByString(value[index++]);
            if(head.left != null) queue.offer(head.left);
            if(head.right != null) queue.offer(head.right);
        }
        return head;

    }


    public Node generateNodeByString(String str){
        if(str.equals("#!"))
            return null;
        return new Node(Integer.valueOf(str));
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
