package algorithm.backTracking;

public class TwoNumberAdd {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null) return null;
        ListNode node = new ListNode(-1);
        ListNode head = node;
        int pre = 0;
        while(l1 != null || l2 != null){
            int num = 0;
            num += pre;
            if(l1 != null){
                num += l1.val;
                l1 = l1.next;
            }
            if(l2 != null){
                num += l2.val;
                l2 = l2.next;
            }
            pre = num/10;
            head.next =  new ListNode(num%10);
            head = head.next;
        }
        head.next = pre != 0? new ListNode(pre):null;
        return node.next;
    }


    public static void main(String args[]){
        ListNode l1 = new ListNode(0);
        ListNode l2 = new ListNode(0);
        ListNode node1 = l1;
        ListNode node2 = l2;
        int[] a = {2,4,3};
        int[] b = {5,6,4};
        for(int i = 0; i < 3; i++){
            node1.next = new ListNode(a[i]);
            node2.next = new ListNode(b[i]);
            node1 = node1.next;
            node2 = node2.next;
        }
        TwoNumberAdd tad= new TwoNumberAdd();
        ListNode node = tad.addTwoNumbers(l1.next, l2.next);
        while (node!=null){
            System.out.println(node.val);
            node = node.next;
        }

    }
}
