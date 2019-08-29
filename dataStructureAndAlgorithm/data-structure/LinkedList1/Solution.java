package LinkedList1;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class Solution {
    /**
     * 环形链表 判断链表是否有环
     * @param head 头节点
     * @return
     */
    public boolean hasCycle(ListNode head){
        Set<ListNode> node = new HashSet<ListNode>();
        while (head != null){
            if(node.contains(head))
                return true;
            else node.add(head);
            head = head.next;
        }
        return false;
    }

    /**
     * 使用快慢指针，判断链表是否有环
     * @param head
     * @return
     */
    public boolean hasCycle1(ListNode head) {
        ListNode slow=head, fast=head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast) return true;
        }
        return false;
    }

    /**
     * 判断链表是否相交
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Set<ListNode> node = new HashSet<ListNode>();
        ListNode A = headA, B = headB;
        while (A != null || B != null){
            if(A != null){
                if(node.contains(A)) return A;
                else {
                    node.add(A);
                    A = A.next;
                }
            }
            if(B != null){
                if(node.contains(B)) return B;
                else {
                    node.add(B);
                    B = B.next;
                }
            }
        }
        return null;
    }
    public ListNode getIntersectionNode1(ListNode headA, ListNode headB) {
        /**
         * 定义两个节点分别向后移动，当移动到末尾的时候，让指针指向另一个链表的头部，
         * 当两个指针相遇的时候，返回当前节点。
         */
        if(headA == null || headB == null ) return null;
        ListNode nodeA = headA, nodeB = headB;
        while(nodeA != nodeB )
        {
            nodeA =nodeA == null? headB : nodeA.next;
            nodeB =nodeB == null? headA : nodeB.next;
        }
        return nodeA;
    }
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB){
        /**
         * 使用栈判断栈顶元素
         */
        Stack<ListNode> stackA = new Stack<ListNode>();
        Stack<ListNode> stackB = new Stack<ListNode>();
        ListNode nodeA = headA, nodeB = headB, temp=null;
        while (nodeA != null){
            stackA.push(nodeA);
            nodeA = nodeA.next;
        }
        while (nodeB != null){
            stackB.push(nodeB);
            nodeB = nodeB.next;
        }
        while (!stackA.isEmpty() && !stackB.isEmpty()){
            if(stackA.peek() == stackB.peek()){
                temp = stackA.pop();
                stackB.pop();
            }else return temp;
        }
        return temp;
    }
}
