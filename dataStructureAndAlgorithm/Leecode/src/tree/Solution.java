package tree;



import javafx.util.Pair;

import java.util.*;

/**
 * 1 相同的树
 * 2 对称二叉树
 * 3 二叉树的最大深度
 * 4 二叉树的层次遍历
 * 5 将有序数组转化为二叉搜索树
 * 6 平衡二叉树
 * 7 二叉树的最小深度
 * 8 路径总和
 * 9 翻转二叉树
 * 10 二叉搜索树的最近公共祖先
 * 11 二叉树的所有路径
 * 12 左叶子之和
 * 13 路径总和III
 * 14 二叉搜索树中的众数
 * 15 二叉搜索树的最小绝对差
 * 16 把二叉搜索树转换为累加树
 * 17 二叉树的直径
 * 18 二叉树的坡度
 * 19 合并二叉树
 * 20 另一个树的子树
 * 21 二叉树的层平均值
 * 22 两数之和IV-输入BST
 * 23 修剪二叉搜索树
 * 24 二叉树中第二小的节点
 * 25 最长同值路径
 * 26 N叉树的层序遍历
 * 27 N叉树的最大深度。
 * 28 N叉树的前序遍历
 * 29 N叉树的后序遍历
 * 30 二叉搜索树中的搜索
 * 31 二叉搜索树节点最小距离
 * 32 叶子相似的树
 * 33 递增顺序查找树
 * 34 二叉搜索树的范围和
 * 35 单值二叉树
 * 36 二叉树的堂兄弟节点
 * 37 从根到叶的二进制数之和
 */
public class Solution {
    /**
     * 给定两个二叉树，编写一个函数来检验它们是否相同
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     */
    public boolean isSameTree(TreeNode p, TreeNode q){
        if(p == null && q == null) return true;
        if (p != null && q != null && p.val == q.val){
            return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
        }else return false;

    }

    /**
     * 如果同时满足下面的条件，两个树互为镜像：
     *   它们的两个根结点具有相同的值。
     *   每个树的右子树都与另一个树的左子树镜像对称。
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root){
        return mirror(root,root);
    }
    public boolean mirror(TreeNode root1, TreeNode root2){
        if(root1 == null && root2 == null) return true;
        if(root1 == null || root2 == null) return false;
        return (root1.val == root2.val) &&
                mirror(root1.left,root2.right)&&
                mirror(root1.right,root2.left);
    }

    /**
     * 给定一个二叉树，找出其最大深度。
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。、
     * 说明：叶子节点是指没有子节点的节点。
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root){
        if(root == null)
        {
            return 0;
        }
        else {
            int leftHeight = maxDepth(root.left);
            int rightHeight = maxDepth(root.right);
            return java.lang.Math.max(leftHeight, rightHeight)+1;
        }
    }

    /**
     *在栈的帮助下将，将递归转化为迭代
     */
    public int maxDepth1(TreeNode root){
        Queue<Pair<TreeNode, Integer>> stack = new LinkedList<>();
        if(root != null){
            stack.add(new Pair<>(root,1));
        }
        int depth = 0;
        while (!stack.isEmpty()){
            Pair<TreeNode, Integer> current = stack.poll();
            root = current.getKey();
            int current_depth = current.getValue();
            if(root != null){
                depth = Math.max(depth,current_depth);
                stack.add(new Pair<>(root.left, current_depth+1));
                stack.add(new Pair<>(root.right,current_depth+1));
            }
        }
        return depth;
    }

    /**
     * 给定一个二叉树，返回其节点值自底向上的逐层遍历。
     * （即按从叶子节点所在层到根节点所在层，逐层从左到右遍历）
     * @param root
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root){

        List<List<Integer>> res = new ArrayList<>();
        helper(root,0,res);
        Collections.reverse(res);
        return res;
    }
    //递归解决
    public void helper(TreeNode root, int level, List<List<Integer>> res){
        if(root == null )return;
        if(level+1 >res.size())
            res.add(new ArrayList<>());
        res.get(level).add(root.val);
        if(root.left != null)helper(root.left,level+1,res);
        if(root.right != null)helper(root.right, level+1, res);
    }

    //迭代解决
    public List<List<Integer>> levelOrderBottom1(TreeNode root){
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                list.add(node.val);
                if(node.left != null) queue.offer(node.left);
                if(node.right != null) queue.offer(node.right);
            }
            res.add(list);
        }
        Collections.reverse(res);
        return res;
    }

    /**
     * 将一个按照升序排列的有序数组，转换为一颗高度平衡的二叉搜索树。
     * 高度平衡二叉树是指一个二叉树每个节点的左右两个子树的高度差的绝对值不超过1.
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums){
        return toBST(nums, 0, nums.length-1);
    }
    public TreeNode toBST(int[] nums,int left, int right){
        if(left > right)return null;
        int middle = (left+right)/2;
        TreeNode root = new TreeNode(nums[middle]);
        if(left < right){
            root.left = toBST(nums,left, middle-1);
            root.right = toBST(nums,middle+1,right);
        }
        return root;
    }

    /**
     *给定一个二叉树，判断他是否是高度平衡的二叉树
     * 一个高度平衡的二叉树定义为：
     *    一个二叉树每个节点的左右两个子树的高度差的绝对值不超过1。
     */
    public boolean isBalanced(TreeNode root){
        if(root == null) return true;
        else {
            if(Math.abs(getDepth(root.left)-getDepth(root.right)) <= 1)
                return true;
            return(isBalanced(root.left) && isBalanced(root.right));
        }
    }
    public int getDepth( TreeNode root){
        if(root == null)return 0;
        return Math.max(getDepth(root.left), getDepth(root.right))+1;
    }

    /**
     *给定一个二叉树，找出其最小深度。
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * 叶子节点是指没有子节点的节点。
     */
    public int minDepth(TreeNode root){
        if(root == null) return 0;
        if(root.left == null && root.right == null)return 1;
        int min_depth = Integer.MAX_VALUE;
        if(root.left != null) min_depth = Math.min(minDepth(root.left), min_depth);
        if(root.right != null) min_depth = Math.min(minDepth(root.right), min_depth);
        return min_depth+1;

    }
    public int minDepth1(TreeNode root){
        LinkedList<Pair<TreeNode, Integer>> stack = new LinkedList<>();
        if(root == null)return 0;
        else stack.add(new Pair<>(root,1));
        int min_depth = Integer.MAX_VALUE;
        while (!stack.isEmpty()){
            Pair<TreeNode, Integer> current = stack.poll();
            root = current.getKey();
            int currentLength = current.getValue();

            if(root.left == null && root.right == null){
                return Math.min(currentLength,min_depth);
            }
            if(root.left != null) stack.add(new Pair<>(root.left, currentLength+1));
            if(root.right != null)stack.add((new Pair<>(root.right, currentLength+1)));
        }
        return min_depth;
    }

    public int minDepth2(TreeNode root){
        if(root == null){return 0;}
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if(left == 0 || right == 0)
            return Math.max(left, right)+1;
        else
            return Math.min(left, right)+1;
    }


    /**
     * 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，
     * 这条路径上所有节点值相加等于目标和。
     * 说明: 叶子节点是指没有子节点的节点
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum){
        if(root == null) return false;
        if(root.left == null && root.right == null){
           return sum == 0;
        }
        return hasPathSum(root.left, sum-root.val)||hasPathSum(root.right,sum-root.val);
    }
    public boolean hasPathSum1(TreeNode root, int sum){
        if(root == null)return false;
        Queue<Pair<TreeNode, Integer>> queue = new LinkedList<>();
        sum-=root.val;
        queue.add(new Pair<>(root, sum));
        while (!queue.isEmpty()){
            for(int i = 0; i < queue.size(); i++){
                Pair<TreeNode, Integer> current = queue.poll();
                TreeNode node = current.getKey();
                sum = current.getValue();
                if(sum == 0 && node.left == null && node.right == null)return true;
                if(node.left != null) queue.add(new Pair<>(node.left, sum-node.left.val));
                if(node.right != null) queue.add(new Pair<>(node.right, sum-node.right.val));
            }
        }
        return false;
    }

    /**
     * 反转二叉树
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root){

        if(root == null) return null;
        TreeNode temp = invertTree(root.right);
        root.right = invertTree(root.left);
        root.left = temp;
//        root.right = left;
//        root.left = right;

        return root;
    }
    public TreeNode invertTree1(TreeNode root){
        if(root == null)return null;
        else{
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            if(root.left != null) invertTree(root.left);
            if(root.right != null) invertTree(root.right);
        }
        return root;
    }

    //通过迭代的方式翻转
    public TreeNode invertTree2(TreeNode root){
        if(root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            TreeNode temp = node.right;
            node.right = node.left;
            node.left = temp;
            if(node.left != null)queue.add(node.left);
            if(node.right != null) queue.add(node.right);
        }
        return root;
    }

    /**
     * 给定一个二叉搜索树，找到该树中两个指定节点的最近公共祖先。
     *
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        if(p.val > root.val && q.val >root.val)return lowestCommonAncestor(root.right,p,q);
        if(p.val < root.val && q.val < root.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }

    public List<String> binaryTreePaths(TreeNode root){
        List<String> lis = new LinkedList<>();
        if(root == null)return lis;
        getPath(lis,Integer.toString(root.val),root);
        return lis;
    }
    public void getPath(List<String> lis, String str, TreeNode root){
        if(root.left == null && root.right == null){
            String temp = str;
            lis.add(temp);
            return;
        }
        str += "->";
        if(root.left != null) getPath(lis, str+Integer.valueOf(root.left.val),root.left);
        if(root.right != null) getPath(lis, str+Integer.valueOf(root.right.val), root.right);
    }

    /**
     * 求所有左叶子节点的和。
     */
    public int sumOfLeftLeaves(TreeNode root){
        int sum = 0;
        if(root == null) return sum;
        sum += getsum(sum,root);
        return sum;
    }
    public int getsum(int sum, TreeNode root){
        if(root.left != null)
        {
            if(root.left.left == null && root.left.right == null)
                sum += getsum(root.left.val,root.left);
            else sum+= getsum(0, root.left);
        }
        if(root.right != null)sum += getsum(0, root.right);
        return sum;
    }

    /**
     * 给定一个二叉树，它的每个节点都存放着一个整数值
     *
     * 找出路径和等于给定数值的路径总数
     *
     * 路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的
     * （只能从父节点到子节点）
     */
    // 第一种方法以每个节点为根节点，计算所有的节点路径
    public int pathSum1(TreeNode root, int sum){
        if(root == null)return 0;
        return helper1(root, sum) + pathSum1(root.left, sum) + pathSum1(root.right, sum);
    }
    int helper1(TreeNode root, int sum){
        if(root == null)return 0;
        sum -= root.val;
        return (sum == 0?1:0) + helper1(root.left, sum) + helper1(root.right, sum);
    }

    //DFS + 回溯

    public int pathSum(TreeNode root, int sum){
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0,1);
        return helper(root, map, sum, 0);
    }

    public int helper(TreeNode root, HashMap<Integer,Integer> map, int sum, int pathSum){
        int res = 0;
        if(root == null)return 0;
        pathSum += root.val;
        res += map.getOrDefault(sum-pathSum, 0);
        map.put(pathSum, map.getOrDefault(pathSum,0)+1);
        res += helper(root.left,map,sum,pathSum) + helper(root.right,map,sum,pathSum);
        map.put(pathSum,map.get(pathSum)-1);
        return res;
    }
    public int pathSum2(TreeNode root, int sum){
        int count = 0;
        if(root == null) return 0;
        if(sum == root.val)
            count++;
        if(root.left != null)
        count += pathSum2(root.left, sum-root.left.val);
        if(root.right != null)
        count += pathSum2(root.right,sum-root.right.val);
        return count;
    }

    /**
     * 给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

     假定 BST 有如下定义：

     结点左子树中所含结点的值小于等于当前结点的值
     结点右子树中所含结点的值大于等于当前结点的值
     左子树和右子树都是二叉搜索树
     */
    TreeNode pre;
    int max = 0;
    int cur = 1;
    public int[] findMode(TreeNode root){
        if(root == null)return new int[] {};
        List<Integer> list = new LinkedList<>();
        inorder(root, list);
        return list.stream().mapToInt(Integer::valueOf).toArray();
    }

    public void inorder(TreeNode root, List<Integer> list){
        if(root == null)return;
        inorder(root.left,list);
        if(pre != null){
            if(pre.val == root.val){
                cur++;
            }else {
                cur = 1;
            }
        }
        if(max == cur){
            list.add(root.val);
        }
        if(cur > max){
            list.clear();
            list.add(root.val);
            max = cur;
        }
        pre = root;
        inorder(root.right,list);
    }
    List<Integer> list = new LinkedList<>();
    public int[] findMode1(TreeNode root){
        if(root == null)return new int[] {};
        midt(root);
        return list.stream().mapToInt(Integer::valueOf).toArray();
    }

    private int preValue = 0;
    private int now = 0;
    private int maxCount = 0;
    private void midt(TreeNode root){
        if(root == null)return;
        midt(root.left);
        if(now == 0){
            now++;
            preValue = root.val;
        }else {
            if(preValue == root.val){
                now++;
            }else {
                preValue = root.val;
                now = 1;
            }
            if(now > maxCount){
                list.clear();
                list.add(preValue);
                maxCount = now;
            }else if(now == maxCount){
                list.add(preValue);
            }
        }
        midt(root.right);
    }

    /**
     * 给定一个多有节点非负值的二叉搜索树，求树中任意两节点的差的绝对值的最小值。
     * @param root
     * @return
     */
    int min = Integer.MAX_VALUE;
    public int getMinimumDifference(TreeNode root){
        preorder(root);
        return min;
    }
    public void preorder(TreeNode root){
        if(root == null) return;
        else {
            if(root.left != null){
                TreeNode left = root.left;
                while (left.right != null)
                {
                    left = left.right;
                }
                int val = Math.abs(left.val - root.val);
                if(min > val){
                    min = val;
                }
            }
            if(root.right != null){
                TreeNode right = root.right;
                while (right.left != null){
                    right = right.left;
                }
                int val = Math.abs(right.val - root.val);
                if(min > val){
                    min = val;
                }
            }
        }
        preorder(root.left);
        preorder(root.right);
    }

    TreeNode preNode;
    public void midorder(TreeNode root){
        if(root == null)return;
        midorder(root.left);
        if(preNode != null){
            int value = Math.abs(preNode.val - root.val);
            if(min > value)min = value;
        }
        preNode = root;
        midorder(root.right);
    }

    /**
     * 给定一个二叉搜索树，把它转化为累加树，使得每个节点的值是原来节点值加上所有大于它的节点值之和
     * @param root
     * @return
     */
    public TreeNode convertBST(TreeNode root){
//        reinorder(root,0);
        return root;
    }
    public int reinorder(TreeNode root, int n){
        if(root == null)return n;
        int rightValue = reinorder(root.right, n);
        root.val += rightValue;
        int leftValue = reinorder(root.left, root.val);
        return leftValue;
    }
    int sum = 0;
    public void remidorder(TreeNode root){
        if(root == null) return;
        remidorder(root.right);
        root.val += sum;
        sum = root.val;
        remidorder(root.left);
    }

    /**
     * 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个节点路径长度中的最大值。
     * 这条路径可能穿过根节点
     * @param root
     * @return
     */

    int res = 0;
    public int diameterOfBinaryTree(TreeNode root){
        maxPathDepth(root);
        return res;
    }

    /**
     *最大的直径可能有三种情况：
     1.在左子树内部
     2.在右子树内部
     3.在穿过左子树，根节点，右子树的一条路径中

     设计一个递归函数，返回以该节点为根节点，向下走的最长路径
     知道这个值以后
     以某个节点为根节点的最长直径就是，该节点左子树向下走的最长路径 ，再加上该节点右子树向下走的最长路径
     我们用一个全局变量记录这个最大值，不断更新
     */
    public int maxPathDepth(TreeNode root){
        if(root == null){
            return 0;
        }
        int l = maxPathDepth(root.left);
        int r = maxPathDepth(root.right);
        res = Math.max(res,(l+r));
        return Math.max(l, r)+1;
    }

    /**
     * 给定一个二叉树，计算整个数的坡度。
     * 一个树的节点坡度定义即为，该节点左子树的节点之和和右子树节点之和的差的绝对值。
     * 空节点的坡度是0.
     * 整个树的坡度就是其所有节点的坡度之和。
     * @param root
     * @return
     */
    int tilt = 0;
    public int findTilt(TreeNode root){
        help(root);
        return tilt;
    }
    public int help(TreeNode root){
        if(root == null)return 0;
        int left_sum = help(root.left);
        int right_sum = help(root.right);
        tilt += Math.abs(left_sum -right_sum);
        return left_sum+right_sum+root.val;
    }

    /**
     * 给定两个二叉树，想想当你将它们中的一个覆盖到另一个时，两个二叉树的一些节点便会重叠。
     * 你需要将他们合并为一个新的二叉树，合并的规则是如果两个节点重叠，那么将他们的值相加作为合并后的新值，
     * 否则不为null的节点将直接作为新二叉树的节点。
     * @param t1
     * @param t2
     * @return
     */
    //两棵树合并新树
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2){
        if(t1 == null && t2 == null)return null;
        TreeNode node = new TreeNode((t1== null?0:t1.val) + (t2== null?0:t2.val));
        node.left = mergeTrees(t1== null?null:t1.left, t2== null?null:t2.left);
        node.right = mergeTrees(t1== null?null:t1.right, t2 == null?null:t2.right);
        return node;
    }
    //t2并在t1上。
    public TreeNode mergeTrees1(TreeNode t1,TreeNode t2){
        if(t2 == null) return t1;
        if(t1 == null) return t2;
        t1.val += t2.val;
        t1.left = mergeTrees( t1.left, t2.left);
        t1.right = mergeTrees(t1.right, t2.right);
        return t1;
    }

    /**
     * 判断两颗树中其中一棵树否是另一颗的子树
     * @param s
     * @param t
     * @return
     */
    public boolean isSubtree(TreeNode s, TreeNode t) {
        return isTheSameTree(s, t) || s != null && (isSubtree(s.left, t) || isSubtree(s.right, t));
    }

    public boolean isTheSameTree(TreeNode a, TreeNode b) {
        if (a != null && b != null) {
            return a.val == b.val && isTheSameTree(a.left, b.left) && isTheSameTree(a.right, b.right);
        }
        return a == null && b == null;
    }
    public boolean isSubtree1(TreeNode s, TreeNode t){
        Queue<TreeNode> queue1 = new LinkedList<>();
        Queue<TreeNode> queue2 = new LinkedList<>();
        TreeNode node = t;
        queue2.add(node);
        queue1.add(s);
        while (!queue1.isEmpty()){
            s = queue1.poll();
            if(!queue2.isEmpty() && queue2.peek().val == s.val)
            {
                node = queue2.poll();
                if(node.left != null) queue2.add(node.left);
                if(node.right != null) queue2.add(node.right);
                if(queue2.isEmpty() && s.left == null && s.right ==null)return true;
                while(!queue1.isEmpty()&& queue1.peek().val != node.val)queue1.remove();
            } else {
                node = t;
                queue2.clear();
                queue2.add(node);
            }
            if(s.left != null) queue1.add(s.left);
            if(s.right != null) queue1.add(s.right);

        }
        return false;
    }

    /**
     * 求二叉树的每层的平均值，并把它们保存在list中，然后返回
     * @param root
     * @return
     */
    public List<Double> averageOfLevels(TreeNode root){
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<Double> list = new LinkedList<>();
        while (!queue.isEmpty()){
            int size = queue.size();
            double sum = 0;
            for(int i = 0; i < size; i++){
                root = queue.poll();
                sum+=root.val;
                if(root.left != null) queue.add(root.left);
                if(root.right != null) queue.add(root.right);
            }
            list.add(sum/size);
        }
        return list;
    }

    /**
     * 给定一个二叉搜索树和一个目标结果，如果BST中存在两个元素且它们的和等于给定目标结果，则返回true。
     * @param root
     * @param k
     * @return
     */
    // hashSet方法
    Set<Integer> set = new HashSet<>();
    public boolean findTarget(TreeNode root,int k){
        if(root == null) return false;
        if(set.contains(k-root.val))
            return true;
        else
            set.add(root.val);
        return findTarget(root.left,k) || findTarget(root.right,k);
    }
    //双指针法
    public boolean findTarget1(TreeNode root, int k){
        List<Integer> list = new LinkedList<>();
        midOrder(list, root);
        for(int i = 0, j = list.size()-1;i < j;){
            int a = list.get(i)+list.get(j);
            if(a == k)return true;
            else if(a > k)j--;
            else i++;
        }
        return false;
    }
    public void midOrder(List<Integer> list, TreeNode root){
        if(root ==null) return;
        midOrder(list, root.left);
        list.add(root.val);
        midOrder(list, root.right);
    }

    //BST查找法
    public boolean findTarget2(TreeNode root, int k){
        return preOrder(root,root,k);
    }

    public TreeNode findV(TreeNode node,int v){
        if(node == null) return null;
        if(node.val == v)return node;
        else if(node.val > v)return findV(node.left, v);
        else return findV(node.right,v);
    }

    public boolean preOrder(TreeNode node, TreeNode root, int k){
        if(node == null)return false;
        int target = k - node.val;
        TreeNode foundNode = findV(root,target);
        return foundNode != null && node != foundNode || preOrder(node.left,root,k) || preOrder(node.right, root, k);
    }


    /**
     * 修剪二叉树，使得所有的节点值都在[L,R]范围内，返回新的子树的根节点
     * @param root
     * @param L
     * @param R
     * @return
     */
    public TreeNode trimBST(TreeNode root, int L, int R){
        if(root ==  null)return null;
        if(root.val < L)return trimBST(root.right, L, R);
        else if(root.val > R)return trimBST(root.left, L, R);
        else {
            root.left = trimBST(root.left,L,R);
            root.right = trimBST(root.right,L,R);
            return root;
        }
    }

    /**
     * 给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。
     * 如果一个节点有两个子节点的话，那么这个节点的值不大于它的子节点的值。
     * 给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。
     * @param root
     * @return
     */
    public int findSecondMinimumValue(TreeNode root){
        return findSecond(root,root.val);
    }
    public int findSecond(TreeNode root, int val){
        if(root == null)return -1;
        if(root.val > val)return root.val;
        int l = findSecond(root.left,val);
        int r = findSecond(root.right,val);
        if(l > root.val && r > root.val)return Math.min(l,r);
        return Math.max(l,r);
    }

    /**
     * 给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。
     * 这条路径可以经过也可以不经过根节点。
     * @param root
     * @return
     */
    int maxVal;
    public int longestUnivaluePath(TreeNode root){
        if(root == null)return 0;
        posOrder(root,root.val);
        return maxVal;
    }

    /**
     *有点不太明白
     */
    public int posOrder(TreeNode root, int val){
        if(root == null) return 0;
        int left = posOrder(root.left,root.val);
        int right = posOrder(root.right, root.val);
        maxVal = Math.max(maxVal, left+right);
        if(root.val == val)return Math.max(left,right)+1;
        return 0;
    }

    /**
     * 给定一个N叉树，返回其节点值的层序遍历。（即从左到右，逐层遍历）。
     * @param root
     * @return
     */
    //迭代
    public List<List<Integer>> levelOrder(Node root){
        List<List<Integer>> res = new LinkedList<>();
        if(root == null) return res;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            List<Integer> list = new LinkedList<>();
            int size = queue.size();
            for(int i = 0; i < size; i++) {
                root = queue.poll();
                list.add(root.val);
                if(root.children.size() != 0){
                    queue.addAll(root.children);
                }
            }
            res.add(list);
        }
        return res;
    }

    //递归
    public List<List<Integer>> levelorder1(Node root){
        List<List<Integer>> res = new LinkedList<>();
        if(root == null) return res;
        myhelper(root, 0, res);
        return res;
    }
    public void myhelper(Node root, int depth, List<List<Integer>> res){
        if(root ==  null) return;
        if(depth + 1 > res.size()){
            res.add(new LinkedList<>());
        }
        res.get(depth).add(root.val);
        for(Node node: root.children){
            if(node!=null){
                myhelper(node,depth+1,res);
            }
        }
    }

    /**
     * 给定一个N叉树，找到其最大深度
     * 最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
     * @param root
     * @return
     */
    //迭代
    public int maxDepth1(Node root){
        if(root == null)return 0;
        int depth = 0;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            while(size-- > 0){
                root = queue.poll();
                queue.addAll(root.children);
            }
            depth++;
        }
        return depth;
    }
    //递归
    public int maxDepth(Node root){
        if(root == null) return 0;
        //求每个子树的最大深度
        int max = 0;
        for(Node node: root.children){
            int depth = maxDepth(node);
            max = Math.max(depth, max);
        }
        return max+1;
    }

    /**
     * 给定一个N叉树，返回其节点的前序遍历。
     * @param root
     * @return
     */
    public List<Integer> preorder(Node root){
        List<Integer> list = new LinkedList<>();
        if(root == null) return list;
        preOrder(root,list);
        return list;
    }
    public void preOrder(Node root, List<Integer> list){
        if(root == null) return;
        list.add(root.val);
        for(Node node: root.children){
            preOrder(node,list);
        }
    }

    /**
     * 给定一个N叉树，返回其节点值的后序遍历
     * @param root
     * @return
     */
    public List<Integer> posorder(Node root){
        List<Integer> list = new LinkedList<>();
        if(root == null)return list;
        posOrder(root,list);
        return list;
    }
    public void posOrder(Node root, List<Integer> list){
        if(root == null)return;
        for (Node node: root.children){
            posOrder(node,list);
        }
        list.add(root.val);
    }

    /**
     * 给定二叉搜索树（BST）的根节点和一个值。你需要在BST中找到节点值等于给定值的节点。
     * 返回以该节点为根的子树。如果节点不存在，则返回NULL。
     * @param root
     * @param val
     * @return
     */
    public TreeNode searchBST(TreeNode root, int val){
        if(root == null) return null;
        if(root.val > val) return searchBST(root.left,val);
        else if(root.val < val) return searchBST(root.right,val);
        else return root;
    }

    /**
     * 给定一个二叉搜索树的根节点root，返回树中任意两节点的差的最小值。
     * @param root
     * @return
     */
    //同第十五题
    int min1 =  Integer.MAX_VALUE;
    public int minDiffInBST(TreeNode root) {
        midOrder(root);
        return min;
    }
    TreeNode preNode1;
    public void midOrder(TreeNode root){
        if(root == null)return;
        midOrder(root.left);
        if(preNode != null){
            int val = Math.abs(preNode.val - root.val);
            if(val < min) min = val;
        }
        preNode = root;
        midOrder(root.right);
    }

    /**
     * 考虑一颗二叉树上所有的叶子，这些叶子的值是按从左到右的顺序排列形成一个叶值序列
     * 如果两颗二叉树的叶值序列相同，那么我们就认为它们是叶相似的。
     * 如果给定的两个头节点分别为root1和root2的树是叶相似的，则返回true;否则返回false.
     */
    public boolean leafSimilar1(TreeNode root1, TreeNode root2){
        if(root1 == null && root2 == null)return true;
        List<Integer> list1 = new LinkedList<>();
        List<Integer> list2 = new LinkedList<>();
        DSF(root1,list1);
        DSF(root2,list2);
        if(list1.size() != list2.size())return false;
        for(int i = 0; i < list1.size(); i++){
            if(!list1.get(i).equals(list2.get(i)))return false;
        }
        return true;
    }

    public void DSF(TreeNode root, List<Integer> list){
        if(root != null){
            if(root.left == null && root.right == null){
                list.add(root.val);
            }
            DSF(root.left,list);
            DSF(root.right,list);
        }
    }

    // BFS 通过栈遍历两颗树，通过队列判断叶节点序列。
    public boolean leafSimilar(TreeNode root1, TreeNode root2){
        Stack<TreeNode> stack = new Stack<>();
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode node1 = root1;
        while(node1 != null || stack.size() > 0){
            while (node1 != null){
                stack.push(node1);
                node1 = node1.left;
            }
            node1 = stack.pop();
            if(node1.left == null && node1.right == null){
                queue.add(node1);
            }
            node1 = node1.right;
        }
        node1 = root2;
        while (node1 != null || stack.size() >0){
            while (node1 != null){
                stack.push(node1);
                node1 = node1.left;
            }
            node1 = stack.pop();
            if(node1.left == null && node1.right == null){
                if(queue.size() > 0){
                    if(queue.peek().val == node1.val)
                        queue.poll();
                    else return false;
                }
            }
            node1 = node1.right;
        }
        return queue.size() == 0;
    }

    /**
     * 给定一个树，按中序遍历重新排列数，使树中最左边的节点现在是树的根，
     * 并且每个节点没有左子节点，只有一个右子节点。
     * @param root
     * @return
     */
    public TreeNode increaseingBST(TreeNode root){
        if(root == null)return null;
        List<Integer> list = new ArrayList<>();
        preOrderTree(root, list);
        TreeNode node = new TreeNode(0);
        root = node;
        while (list.size() > 0){
            node.right = new TreeNode(list.get(0));
            node = node.right;
            list.remove(0);
        }
        return root.right;
    }
    public void preOrderTree(TreeNode root, List<Integer> list){
        if(root == null)return;
        preOrderTree(root.left, list);
        list.add(root.val);
        preOrderTree(root.right, list);
    }

    /**
     *################################################################################################################
     *                                              动态规划示例
     * ###############################################################################################################
     */
    //动态规划
    public TreeNode increaseingBST(TreeNode root, TreeNode next){
        if(root == null)return null;
        //动态规划：对当前root节点，先处理右子树，再处理左子树
        if(root.right == null)
            //如果当前root没有右子树，直接把next作为其右子树的根。
            root.right = next;
        else
            //如果当前root有右子树，则把右子树中的最左节点作为其右子树的根。
            root.right = increaseingBST(root.right,next);
        //如果处理完右子树，发现没有左子树（即当前root不用作为谁的下一个节点），直接返回root
        if(root.left == null)
            return root;
        //如果有左子树，则把当前root作为其左子树的下一个节点。
        next = root;
        //以当前root作为其左子树的下一个节点，处理左子树。
        TreeNode node = increaseingBST(root.left,next);
        //以当前的root的左右子树都处理完后，形成一条右数链（每个节点最多只有右子树），将当前root节点的左子树置空。
        root.left = null;
        return node;
    }

    /**
     * 给定二叉搜索树的根节点root,返回L和R之间的所有节点的值的和
     * 二叉搜索树保证具有唯一的值。
     * @param root
     * @param L
     * @param R
     * @return
     */
    public int rangeSumBST1(TreeNode root, int L, int R){
        if(root == null) return 0;
        if(root.val < L)
            return rangeSumBST(root.right,L,R);
        if(root.val > R)
            return rangeSumBST(root.left,L,R);
        return root.val+rangeSumBST(root.left,L,R)+rangeSumBST(root.right,L,R);
    }

    public int rangeSumBST(TreeNode root, int L, int R){
        if(root == null)return 0;
        int sum = 0;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            root = stack.pop();
            if(root.val >= L && root.val <= R)
                sum+=root.val;
            if(root.left != null)stack.push(root.left);
            if(root.right != null)stack.push(root.right);
        }
        return sum;
    }

    /**
     * 如果二叉树每个节点都具有相同的值，那么该二叉树就是单值二叉树。
     * ############################################################################################################
     *                                            分治策略
     * ############################################################################################################
     *
     */
    public boolean isUnivalTree(TreeNode root){
        if(root == null)return true;
        boolean left = root.left == null || root.val == root.left.val&&isUnivalTree(root.left);
        boolean right =root.right == null|| root.val == root.right.val && isUnivalTree(root.right);
        return left && right;
    }

    /**
     * 在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。
     * 如果二叉树的两个节点深度相同，但父节点不同，则它们是一对堂兄弟节点。
     * 我们给出了具有唯一值的二叉树的根节点 root，以及树中两个不同节点的值 x 和 y。
     * 只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true。否则，返回 false。
     */
    Map<Integer,Integer> depth;
    Map<Integer,TreeNode> parent;
    public boolean isCousins(TreeNode root, int x, int y){
        depth = new HashMap<>();
        parent = new HashMap<>();
        dfs(root,null);
        return depth.get(x) == depth.get(y) && parent.get(x) != parent.get(y);
    }

    public void dfs(TreeNode root, TreeNode par){
        if(root != null){
            depth.put(root.val, par != null? 1+depth.get(par.val): 0);
            parent.put(root.val, par);
            dfs(root.left,root);
            dfs(root.right,root);
        }
    }

    /**
     * 给出一颗二叉树，其上每个节点的值都是0或1，
     * 每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。
     * @param root
     * @return
     */
    int binarySum = 0;
    public int sumRootToLeaf(TreeNode root){
        if(root == null) return 0;
        preOrderTree(root,"");
        return binarySum;
    }

    public void preOrderTree(TreeNode root, String res){
        if(root == null)return;
        res += root.val;
        if(root.left == null && root.right == null){

            binarySum+=Integer.parseInt(res,2);
        }
        preOrderTree(root.left, res);
        preOrderTree(root.right,res);
    }
    public int sumRootToLeaf1(TreeNode root){
        if(root == null)return 0;
        callLeaf(root,0);
        return binarySum;

    }

    public void callLeaf(TreeNode root, int curNum){
        if(root == null)return;
        int newSum = (curNum << 1) + root.val;
        if(root.left == null && root.right == null){
            binarySum += newSum;
        }
        callLeaf(root.left,newSum);
        callLeaf(root.right,newSum);
    }
    public static void main(String[] args){
        int[] treeVal = new int[]{5,4,8,11,0,13,4,7,2,0,0,0,1};
        TreeNode node = new TreeNode(0);
        node = new TreeNode(9);
        List<TreeNode> nodeList = new LinkedList<TreeNode>();
        for(int i = 0;i< treeVal.length; i++){
            nodeList.add(treeVal[i]!= 0?new TreeNode(treeVal[i]):null);
        }
        for(int parentIndex = 0; parentIndex < treeVal.length/2-1;parentIndex++){
            nodeList.get(parentIndex).left = nodeList.get(parentIndex*2+1);
            nodeList.get(parentIndex).right = nodeList.get((parentIndex+1) *2);
        }
        int lastParentIndex = treeVal.length/2-1;
        nodeList.get(lastParentIndex).left = nodeList.get(lastParentIndex*2+1);
        if(treeVal.length %2 == 0){
            nodeList.get(lastParentIndex).right = nodeList.get((lastParentIndex+1)*2);
        }


    }
}
