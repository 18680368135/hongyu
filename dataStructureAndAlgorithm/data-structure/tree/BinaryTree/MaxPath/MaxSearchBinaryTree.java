package tree.BinaryTree.MaxPath;

public class MaxSearchBinaryTree {

    public ReturnType process(Node X){
        if(X == null) return new ReturnType(null, 0, Integer.MIN_VALUE, Integer.MIN_VALUE);

        ReturnType lData = process(X.left);
        ReturnType rData = process(X.right);
        int min = Math.min(X.value, Math.min(lData.min, rData.min));
        int max = Math.max(X.value, Math.max(lData.max, rData.max));
        int maxBSTSize = Math.max(lData.maxBSTSize, rData.maxBSTSize);

        Node maxBSTHead = lData.maxBSTSize >= rData.maxBSTSize ? lData.maxBSTHead : rData.maxBSTHead;

        if(lData.maxBSTHead == X.left && rData.maxBSTHead == X.right && X.value > lData.max && X.value < rData.min){

            maxBSTSize = lData.maxBSTSize + rData.maxBSTSize + 1;
            maxBSTHead = X;

        }
        return new ReturnType(maxBSTHead, maxBSTSize, min, max);
    }

    public Node getMaxBST(Node head){
        return process(head).maxBSTHead;
    }

}


class ReturnType{
     public Node maxBSTHead;
     public int maxBSTSize;
     public int min;
     public int max;


     public ReturnType(Node maxBSTHead, int maxBSTSize, int min, int max){
         this.maxBSTHead = maxBSTHead;
         this.maxBSTSize = maxBSTSize;
         this.min = min;
         this.max = max;

     }

}