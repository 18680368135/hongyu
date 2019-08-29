package Queue;

import Interface.IQueue;
import Interface.LNode;


public class LinkedQueue<T> implements IQueue {


    private LNode Header; //头指针，指向头节点，即队首元素的前一个位置（方便删除队首元素）
    private LNode Rear;  //尾指针，方便插入元素
    private Integer size;


    public IQueue InitQueue() {
        if(Header == null){
            Header = new LNode<T>();
            Rear = Header;
            size = 0;
        }
        return this;
    }

    public IQueue DestroyQueue() {
        Header = null;
        Rear = Header;
        size = 0;
        return this;
    }

    public IQueue ClearQueue() {

        Rear = Header;
        Header.setNext(null);
        size = 0;
        return this;
    }

    public Boolean IsEmpty() {
        if(Rear == Header){
            return Boolean.TRUE;
        }
        else {
            return Boolean.FALSE;
        }
    }

    public Integer GetQueueLength() {
        return size;
    }

    public Object GetHead() {
        return (T) Header.getNext().getData();
    }

    public Boolean EnQueue(Object e) {
        LNode newNode = new LNode<T>((T) e, null);
        Rear.setNext(newNode);
        Rear = newNode;
        size++;
        return Boolean.TRUE;
    }

    public T DeleteQueue() {

        if(Header == null){
            System.out.println("队列被销毁");
            return null;
        }
        else if(Header.getNext() == null ){
            System.out.println("队列已空");
            return null;
        }else if(Header.getNext() == Rear){
            T e = (T) Header.getNext().getData();
            Header.setNext(null);
            Header = Rear;
            size--;
            return e;
        }else{
        T e = (T) Header.getNext().getData();
        Header.setNext(Header.getNext().getNext());
        size--;
        return e; }
    }

    public static void main(String args[]){
        LinkedQueue<Integer> LinkedQueue = new LinkedQueue<Integer>();
        LinkedQueue.InitQueue();
        LinkedQueue.EnQueue(1);
        LinkedQueue.EnQueue(2);
        LinkedQueue.EnQueue(3);
        Integer size = LinkedQueue.size;
        System.out.println(LinkedQueue.GetHead());
        LinkedQueue.DestroyQueue();

        for(int i = 0; i < size; i++){
            //System.out.println(LinkedQueue.GetQueueLength());
            System.out.println(LinkedQueue.DeleteQueue());

        }
        System.out.println(LinkedQueue.size);
        System.out.println(LinkedQueue.IsEmpty());


    }
}
