package stack;

import java.util.LinkedList;
import java.util.Queue;

/**
 * 使用队列实现栈的下列操作：

 push(x) -- 元素 x 入栈
 pop() -- 移除栈顶元素
 top() -- 获取栈顶元素
 empty() -- 返回栈是否为空


 */
public class MyStack {
    /** Initialize your data structure here. */
    Queue<Integer> q1;
    Queue<Integer> q2;
    private int top;
    public MyStack() {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        q1.add(x);
        top = x;
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        while (q1.size() >1){
            top = q1.poll();
            q2.add(top);
        }
        int topval = q1.poll();
        Queue<Integer> temp = q1;
        q1 = q2;
        q2 = temp;
        return topval;
    }

    /** Get the top element. */
    public int top() {
        if(q1.size() > 0)
            return top;
        throw new RuntimeException("栈为空，无法出栈");
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return q1.isEmpty();
    }
}
