package stack;

import java.util.Stack;

/**
 * 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

 push(x) -- 将元素 x 推入栈中。
 pop() -- 删除栈顶的元素。
 top() -- 获取栈顶元素。
 getMin() -- 检索栈中的最小元素。
 */
public class MinStack {

    private Stack<Integer> data;
    private Stack<Integer> helper;

    public MinStack(){
        data = new Stack<>();
        helper = new Stack<>();
    }
    // 思路一：数据栈和辅助栈在任何时候都同步。
    public void push(int x){
        data.add(x);
        if(helper.isEmpty() || helper.peek() >= x){
            helper.add(x);
        }else helper.add(helper.peek());
    }
    //思路二：辅助栈和数据栈不同步

    /**
     * 辅助栈的元素为空的时候，必须放入新进来的数
     * 新来的数小于等于辅助栈栈顶元素的时候，才放入
     * 出栈的时候，辅助栈的栈顶元素等于数据栈的栈顶元素，才出栈
     */
    public void push1(int x){
        data.add(x);
        if(helper.isEmpty() || helper.peek() >= x){
            helper.add(x);
        }
    }
    public void pop() {
        if(!data.isEmpty()){
            helper.pop();
            data.pop();
        }
    }

    public void pop1(){
        if(!data.isEmpty()){
            int top = data.pop();
            if(top == helper.peek()){
                helper.pop();
            }
        }
    }
    public int top() {
        if(!data.isEmpty()){
            return data.peek();
        }
        throw new RuntimeException("栈中元素为空，次操作非法");
    }

    public int getMin() {
        if(!helper.isEmpty()){
            return helper.peek();
        }
        throw new RuntimeException("栈中元素为空，次操作非法");
    }
}
