package Interface;

/**
 *
 *定义链表的节点数据类型及节点类型
 * @param <T>
 */
public class LNode<T> {
    private T data;
    private LNode next;

    /**
     * 定义节点类的构造函数
     */
    public LNode(){

    }

    /**
     * 定义链表的指定的构造函数
     */

    public LNode(T data, LNode next){
        this.data = data;
        this.next = next;

    }

    /**
     * 返回得到的数据
     * @return
     */
    public T getData(){
        return data;
    }

    /**
     * 数据赋值
     */

    public void setData(T data){
        this.data = data;
    }

    /**
     * 得到下一个节点
     */
    public LNode getNext(){
        return next;
    }

    /**
     * 设置下一个节点
     */
    public void setNext(LNode next){
        this.next = next;
    }


}
