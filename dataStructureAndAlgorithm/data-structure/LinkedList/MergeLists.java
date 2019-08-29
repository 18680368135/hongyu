package LinkedList;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Stack;

public class MergeLists {
    ListNode head;

    public void addNode(ListNode node){
        ListNode temp = head;
        while(temp.next != null)
            temp = temp.next;
        temp.next = node;
    }
    public int length(){
        int i = 0;
        ListNode temp = head;
        while (temp.next != null)
        {
            temp = temp.next;
            i++;
        }
        return i;
    }
    public void insertNodeByIndex(int index, ListNode node){
        if(index <1 || index > length()+1){
            System.out.println("插入位置不合法");
            return;
        }
        int length = 1;
        ListNode temp = head;
        while(temp.next != null){
            if(index == length++){
                node.next = temp.next;
                temp.next = node;
                return;
            }
            temp = temp.next;
        }
    }

    /**
     * 通过indx删除节点
     * @param index
     */
    public void delNodeByIndex(int index){
        if(index < 1 || index > length()+1){
            System.out.println("给定的位置不合适");
            return;
        }
        int length = 1;
        ListNode temp = head;

        while(temp.next != null){
            if(index == length++){
                temp.next = temp.next.next;
                return;
            }
            temp = temp.next;
        }
    }

    /**
     * 选择排序
     * 对链表中的结点进行排序，按照从小到大的顺序，使用选择排序。
     *    使用双层遍历。第一层遍历，正常遍历链表，第二层遍历，遍历第一层遍历时所用的结点后面所有结点并与之比较
     *    选择排序比较简单，明白其原理，就能够写的出来。
     */
    public void selectSortNode(){
        if(length() < 2){
            System.out.println("无需进行排序");
            return;
        }
        ListNode temp = head;
        while (temp.next != null){
            ListNode secondNode = temp.next;
            while(secondNode.next != null){
                if(temp.next.val > secondNode.next.val){
                    int val = secondNode.next.val;
                    secondNode.next.val = temp.next.val;
                    temp.next.val = val;
                }
                secondNode = secondNode.next;
            }
            temp = temp.next;
        }
    }

    /**
     * 对链表进行插入排序，按从大到小的顺序，只要这里会写，那么手写用数组插入排序
     *    也是一样的。先要明白原理。什么是插入排序，这样才好写代码。
     *    插入排序：分两组，一组当成有序序列，一组当成无序，将无序组中的元素与有序组中的元素进行比较
     *    (如何比较，那么就要知道插入排序的原理是什么这里不过多阐述)
     *        这里我想到的方法是，构建一个空的链表当成有序序列，而原先的旧链表为无序序列，按照原理，一步步进行编码即可
     */
    public void insertSortNode(){

        if(length() < 2){
            System.out.println("无需排序");
            return;
        }
        //创建新链表
        ListNode newHead = new ListNode(0);
        ListNode newTemp = newHead;
        ListNode temp = head;

        if(newTemp.next == null){
            ListNode node = new ListNode(temp.next.val);
            newTemp.next = node;
            temp = temp.next;
        }
        while(temp.next != null){
            while(newTemp.next != null){
                if(newTemp.next.val < temp.next.val){
                    ListNode node = new ListNode(temp.next.val);
                    node.next = newTemp.next;
                    newTemp.next = node;
                    break;
                }
                newTemp = newTemp.next;
            }
            if(newTemp.next == null){
                ListNode node = new ListNode(temp.next.val);
                newTemp.next = node;
            }
            temp = temp.next;
            newTemp = newHead;
        }
        head = newHead;
    }

    public ListNode bubbleSortNode(ListNode head){
        if(head == null || head.next == null){
            return head;
        }
        ListNode temp = head;
        ListNode tail = null;
        while(temp.next != tail){
            while (temp.next != tail){
                if(temp.next.val > temp.next.next.val){
                    int t = temp.next.val;
                    temp.next.val = temp.next.next.val;
                    temp.next.next.val = t;
                }
                temp = temp.next;
            }
            tail = temp;
            temp = head;
        }
        return head;
    }

    public ListNode quickSortNode(ListNode head, ListNode tail){
        if(head == null || head.next == null){
            return head;
        }
        return head;
    }

    /**
     * 选择排序合并两个单链表
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2){
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        ListNode temp = l1;
        ListNode cur = l2;
        while (temp != null){
            while (cur.next != null){
                if(temp.val >= cur.val && temp.val <= cur.next.val){
                    ListNode node = new ListNode(temp.val);
                    node.next = cur.next;
                    cur.next = node;
                    break;
                }
                cur = cur.next;
            }
            if(cur.next == null){
                ListNode node = new ListNode(temp.val);
                if(cur.val < temp.val)
                    cur.next = node;
                else {
                    node.next = l2;
                    l2 = node;
                    cur = node;
                }

            }
            temp = temp.next;
        }
        return l2;
    }

    /**
     * 归并排序合并两个单链表
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists1(ListNode l1, ListNode l2){
        ListNode start = new ListNode(0);
        ListNode cur = start;
        while (l1 != null && l2 != null){
            if(l1.val > l2.val){
                cur.next = l2;
                cur = cur.next;
                l2 = l2.next;
            }else {
                cur.next = l1;
                cur = cur.next;
                l1 = l1.next;
            }
        }
        if(l1 == null){
            cur.next = l2;
        }else {
            cur.next = l1;
        }
        return start.next;
    }

    /**
     * 删除排序链表中的重复元素
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head){
        if(head == null || head.next == null)
            return head;
        ListNode cur = head;
        while(cur.next != null){
            if(cur.val == cur.next.val){
                cur.next = cur.next.next;
                continue;
            }
            cur = cur.next;
        }
        return head;
    }

    /**
     * 删除链表中等于给定值 val 的所有节点。
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val){
        /**
         * 通过创建前一个节点实现
         */
        ListNode node = head, prenode = null;
        while (node != null){
            if(node.val == val){
                if(node == head){
                    head = head.next;
                    node = node.next;
                    continue;
                }else{
                    prenode.next = node.next;
                    node = prenode.next;
                    continue;
                }
            }
            prenode = node;
            node = node.next;
        }
        return head;
    }

    public ListNode removeElements1(ListNode head, int val){
        ListNode header = new ListNode(0);
        header.next = head;
        ListNode cur = header;
        while (cur.next != null){
            if(cur.next.val == val){
                cur.next = cur.next.next;
            }else cur = cur.next;
        }
        return header.next;
    }
    /**
     * 反转一个单链表
     */
    public ListNode reverseList(ListNode head) {
        if(head == null) return head;
        ListNode cur = new ListNode(0);
        ListNode node = head;
        cur.next = node;
        head = head.next;
        while (head != null){
            node.next = head.next;
            head.next = cur.next;
            cur.next = head;
            head = node.next;
        }
        return cur.next;
    }
    /**
     * 使用栈实现链表的反转
     */
    public ListNode reverseList1(ListNode head){
        if(head == null) return null;
        Stack<ListNode> stack = new Stack<>();
        ListNode node = null ;

        while (head != null){
            stack.push(head);
            node = head;
            head = head.next;
        }
        head = node;
        while (!stack.isEmpty()){
            head.next = stack.pop();
            head = head.next;
            if(stack.isEmpty()) {
                head.next = null;
            }
        }
        return node;
    }

    /**
     * 请判断一个链表是否为回文链表
     */
    public boolean isPalindrome(ListNode head){
        if(head == null) return true;
        Deque<ListNode> q = new ArrayDeque<>();
        while (head != null){
            q.add(head);
            head = head.next;
        }
        while (!q.isEmpty()){
            if(q.size() >1){
                if(q.getFirst().val == q.getLast().val){
                    q.removeFirst();
                    q.removeLast();
                }else return false;
            }else return true;
        }
        return true;
    }

    /**
     *使用快慢指针，链表的后半段需要翻转
     */
    public boolean isPalindrome1(ListNode head){
        if(head == null || head.next == null) return true;
        ListNode slow = head, fast = head;
        while(fast.next != null && fast.next.next != null){
           slow = slow.next;
           fast = fast.next.next;
        }
        slow = reverse(slow.next);
        while (slow != null){
            if(slow.val != head.val){
                return false;
            }
            slow = slow.next;
            head = head.next;
        }
        return true;
    }

    public ListNode reverse(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode cur = head, pre = null;
        while (cur != null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    public ListNode reverse1(ListNode head){
        if(head.next == null)
            return head;
        ListNode node = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }

    /**
     *删除某个链表中给定的（非末尾）节点
     */
    public void deleteNode(ListNode node){
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * 寻找链表中的中间节点
     * @param head 头节点属于链表的第一个节点
     * @return
     */
    public ListNode middleNode(ListNode head){
        if(head.next == null ) return head;
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public static void main(String args[]){
         int[] a = new int[]{1,2};
         int[] b = new int[]{5,7};
         ListNode l1 = new ListNode(1);
         ListNode l2 = new ListNode(0);
         ListNode cur1 = l1;
         ListNode cur2 = l2;
        for (int i=1; i < a.length; i++) {
            cur1.next = new ListNode(a[i]);
//            cur2.next = new ListNode(b[i]);
            cur1 = cur1.next;
//            cur2 = cur2.next;
        }
        MergeLists mer = new MergeLists();
        l1 = mer.reverseList1(l1);
        while (l1 != null){
            System.out.print(l1.val + " ");
            l1 = l1.next;
        }
    }


}
