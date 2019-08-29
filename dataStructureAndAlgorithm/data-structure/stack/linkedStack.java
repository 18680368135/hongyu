package stack;

public class linkedStack<T>{
    class Node{
        private Node next;
        public T value;

    }

    Node top = null;

    /**
     * 栈的初始化，初始化一个空节点，然后top指向这个节点
     */
    public void initStack(){

        Node node= new Node();
        node.next = null;
        node.value = null;
        top = node;



    }

    /**
     * 出栈操作 push
     * @param value
     */
    public void pushStack(T value){
        Node node = new Node();
        node.value = value;

        if(top.next == null){
            top.next = node;
        }else{
            node.next = top.next;
            top.next = node;
        }


    }

    /**
     * 入栈操作pop
     */
    public void PopStack(){
        if(top.next == null)
        {
            System.out.println("栈已空，无法出栈");

        }
        System.out.println("出栈元素为 ：" + top.next.value);
        top.next = top.next.next;

    }

    /**
     * 获取栈顶元素
     * @return
     */
    public Object getPeekValue(){
        if(top.next == null){
            System.out.println("栈已空，无法获得栈顶元素");
            return -1;
        }
        return top.next.value;
    }

    /**
     * 判断栈是否为空
     * @return
     */
    public Boolean IsEmpty(){
        if(top.next == null){
            return Boolean.TRUE;
        }
        return Boolean.FALSE;
    }


    /**
     * 获取栈的大小
     * @return
     */
    public int getStackSize(){
        Node tmp = top;
        int i = 0;
        while(tmp.next != null){
            i++;
            tmp = tmp.next;
        }
        return i;
    }

    /**
     * 打印栈中目前存在的元素值
     */
    public void printStackElement(){
        Node tmp = top;
        if(tmp.next == null){
            System.out.println("栈空了");
        }else{
            while(tmp.next != null){
                System.out.print(tmp.next.value + " ");
                tmp = tmp.next;
            }
        }
    }

    public static void main(String args[]){
        linkedStack<String> stack= new linkedStack<String>();
        stack.initStack();
        stack.pushStack("apple");
        stack.pushStack("peer");
        stack.pushStack("apple");
        stack.pushStack("orange");
        stack.pushStack("fruit");
        stack.printStackElement();
        System.out.println(stack.getStackSize());

        System.out.println(stack.getPeekValue());
        stack.PopStack();
        stack.printStackElement();
        stack.PopStack();
        stack.PopStack();
        stack.PopStack();
        stack.PopStack();
        System.out.println(stack.IsEmpty());

    }
}
