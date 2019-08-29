package Interface;

public interface LStack<T> {

    /**
     * 初始化栈
     */
    public void InitStack(T t);

    /**
     * 判断栈是否为空
     */
    public void IsEmpty();

    /**
     * 入栈操作push
     */
    public T PushStack(T t);

    /**
     * 出栈操作pop
     */
    public T PopStack(T t);

    /**
     * 判断栈是否为满
     */
    public void IsFull();
}
