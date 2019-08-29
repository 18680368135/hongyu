package Interface;

public interface IQueue<T> {
    /**
     * 初始化队列，构造一个空队列
     */

    IQueue InitQueue();

    /**
     * 销毁队列
     */
    IQueue DestroyQueue();

    /**
     * 清空队列
     */
    IQueue ClearQueue();

    /**
     * 判断队列是否为空
     */
    Boolean IsEmpty();

    /**
     * 返回队列长度
     */
    Integer GetQueueLength();

    /**
     * 返回队头元素
     */
    T GetHead();

    /**
     * 插入队尾元素，即入队
     */
    Boolean EnQueue(T e);

    /**
     * 删除队头元素，即出队
     */
    T DeleteQueue();
}
