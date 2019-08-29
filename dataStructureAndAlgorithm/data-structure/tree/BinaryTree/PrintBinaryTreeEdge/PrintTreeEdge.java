package tree.BinaryTree.PrintBinaryTreeEdge;

public class PrintTreeEdge {

    public void printEdge1(Node head){
        if(head == null){
            return;
        }
        int height = getHeight(head, 0);
        Node[][] edgeMap = new Node[height][2];

        setEdgeMap(edgeMap, 0, head);

        //打印左边界点
        for(int i = 0; i != edgeMap.length; i++){
            System.out.print(edgeMap[i][0].value + " ");
        }
        // 打印既不是左边界点，又不是又边界点的节点
        printLeafNotInEdgeMap(head, edgeMap, 0);

        //打印右边界点
        for(int i = edgeMap.length-1; i != -1; i--){

            if(edgeMap[i][0] != edgeMap[i][1]){
                System.out.print(edgeMap[i][1].value +" ");
            }
        }


    }


    public int getHeight(Node head, int l){
        if(head == null) return l;

        return Math.max(getHeight(head.left, l+1),getHeight(head.right, l+1));
    }

    public void setEdgeMap(Node[][] edgeMap, int l, Node head){
        edgeMap[l][0] = edgeMap[l][0] == null ? head: edgeMap[l][0];
        edgeMap[l][1] = head;

        setEdgeMap(edgeMap, l+1, head.left);
        setEdgeMap(edgeMap, l+1, head.right);

    }

    public void printLeafNotInEdgeMap(Node head, Node[][] edgeMap, int l){
        if(head == null) return;
        if(head.value != edgeMap[l][0].value && head.value != edgeMap[l][1].value && head.left == null && head.right == null){
            System.out.print(head.value +" ");
        }
        printLeafNotInEdgeMap(head.left, edgeMap, l+1);
        printLeafNotInEdgeMap(head.right, edgeMap, l+1);
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