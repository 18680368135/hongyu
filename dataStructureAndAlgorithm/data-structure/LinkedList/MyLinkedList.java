package LinkedList;

public class MyLinkedList {
    ListNode head;
    int length;
    /** Initialize your data structure here. */
    public MyLinkedList() {
         length = 0;
         head = new ListNode(-1);
    }

    /** Get the value of the index-th node in the linked list.
     * If the index is invalid, return -1.
     **/
    public int get(int index) {
        if(index >=length || index < 0) return -1;
        ListNode node = head;
        for(int i = 0; i <= index; i++){
            node = node.next;
        }
        return node.val;
    }

    /** Add a node of value val before the first element of the linked list.
     *  After the insertion, the new node will be the first node of the linked
     *  list.
     *  */
    public void addAtHead(int val) {
        ListNode node = new ListNode(val);
        node.next = head.next;
        head.next = node;
        length++;
    }

    /** Append a node of value val to the last element of the linked list.
     **/
    public void addAtTail(int val) {
        ListNode node = new ListNode(val);
        ListNode header = head;
        while (header.next != null){
            header = header.next;
        }
        header.next = node;
        length++;
    }

    /** Add a node of value val before the index-th node in the linked list.
     *  If index equals to the length of linked list, the node will
     *  be appended to the end of linked list.
     *  If index is greater than the length, the node will not be inserted.
     **/
    public void addAtIndex(int index, int val) {
        if(index > length)return;
        if(index == length){
            addAtTail(val);
            return;
        }
        if(index < 0){
            index = index + length + 1;
        }
        ListNode header = head;
        ListNode node = new ListNode(val);
        for(int i = 0; i < index; i++){
            header = header.next;
        }
        node.next = header.next;
        header.next = node;
        length++;
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        if(index >= length || index < 0)
            return;
        ListNode node = head;
        for(int i = 0; i < index; i++){
            node = node.next;
        }
        node.next = node.next.next;
        length--;
    }

    public static void main(String args[]){
        MyLinkedList obj = new MyLinkedList();
        int param_1 = obj.get(0);
        obj.addAtHead(12);
        obj.addAtTail(45);
        obj.addAtIndex(1,62);
        obj.deleteAtIndex(1);
    }
}
