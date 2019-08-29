package array;

import java.util.*;

public class Solution {
    /**
     * 1 删除排序数组中的重复项
     * 2 盛最多水的容器
     * 3 三数之和
     * 4 最接近的三数之和
     * 5 四数之和
     * 6 移除元素
     * 7 下一个排列
     * 8 搜索旋转排序数组
     * 9 在排序数组中查找元的第一个和最后一个位置
     * 10 搜索插入位置
     * 11 组合总和
     * 12 组合总和II
     * 13 旋转图像
     * 14 螺旋矩阵
     * 15 跳跃游戏
     * 16 合并区间
     * 17 螺旋矩阵II
     * 18 不同路径
     * 19 不同路径II
     * 20 最小路径和
     * 21 加一
     * 22 矩阵置零
     */

    /**
     * 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
     * 不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
     *
     */

    public int removeDuplicates(int[] nums){
        if(nums == null)return 0;
        int index = 0;
        for(int i = 1; i < nums.length; i++){
            if(index < i && nums[i-1] != nums[i]){
                index++;
                nums[index] = nums[i];
            }
        }
        return index+1;
    }

    /**
     * 给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
     * 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。
     * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     */
    public int maxArea1(int[] height){
        int max = 0;
        for(int i = 0;i < height.length; i++){
            for(int j = i+1; j < height.length; j++){
                int area = (j-i) * (Math.min(height[j] ,height[i]));
                if(max < area)max = area;
            }
        }
        return max;
    }
    // 双指针法
    public int maxArea(int[] height){
        int i = 0, j = height.length-1;
        int max = 0;
        while (i < j){
            max = Math.max(max, (j-i) * Math.min(height[j], height[i]));
            if(height[j] > height[i]) i++;
            else j--;
        }
        return max;
    }

    /**
     * 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
     * 使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
     */
    /**
     *
     如果Brute force，则是O(n3)时间复杂度，有优化空间。
     先将给定nums排序，简化问题，复杂度为O(nlogn)。
     令nums[k] + nums[i] + nums[j] == 0，找所有的组合的思路是：遍历三个数字中最左数字的指针k，
     找到数组中所有不重复k对应所有b c组合，即每指向新的nums[k]，都通过双指针法找到所有和为0的nums[i] nums[j]并记录：

     当nums[k] > 0时，直接跳出，因为j > i > k，所有数字大于0，以后不可能找到组合了；
     当k > 0 and nums[k] == nums[k - 1]，跳过此数字，因为nums[k - 1]的所有组合已经被加入到结果，如果本次搜索，只会搜索到重复组合。
     i, j分设在[k, len(nums)]两端，根据sum与0的大小关系交替向中间逼近，如果遇到等于0的组合则加入res中，需要注意：
     移动i j需要跳过所有重复值，否则重复答案会被计入res。

     整体算法复杂度O(n2)。
     */
    public List<List<Integer>> threeSum(int[] nums){
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for(int k = 0; k < nums.length-2; k++){
            if(nums[k] > 0)break;
            if(k > 0 && nums[k] == nums[k-1])continue;
            int i = k+1;
            int j = nums.length-1;
            while (i < j){
                int sum = nums[k] + nums[i] + nums[j];
                if(sum > 0) {
                    while (i < j && nums[j] == nums[--j]);
                } else if(sum < 0) {
                    while (i < j && nums[i] == nums[++i]);
                } else{
                    List<Integer> lis = new ArrayList<Integer>(Arrays.asList(nums[k], nums[i], nums[j]));
                    res.add(lis);
                    while (i < j && nums[j] == nums[--j]);
                    while (i < j && nums[i] == nums[++i]);
                }
            }
        }
        return res;
    }

    /**
     * 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。
     * 找出 nums 中的三个整数，使得它们的和与 target 最接近。
     * 返回这三个数的和。假定每组输入只存在唯一答案。
     */

    // 暴力求解
    public int threeSumClosest(int[] nums,int target){
        Arrays.sort(nums);
        int roundSum = nums[0] + nums[1] + nums[2];
        for(int k = 0; k < nums.length; k++){
            int i = k+1;
            int j = nums.length-1;
            while (i < j){
                int val = nums[k] + nums[i] + nums[j];
                if (Math.abs(target - roundSum) > Math.abs(target - val)) {
                    roundSum = val;
                }
                if(target > val){
                    i++;
                }
                else if(target < val){
                    j--;
                }else return roundSum;
            }
        }
        return roundSum;
    }

    /**
     * 给定一个包含 n 个整数的数组 nums 和一个目标值 target，
     * 判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值
     * 与 target 相等？找出所有满足条件且不重复的四元组。
     */
    public List<List<Integer>> fourSum(int[] nums, int target){
        List<List<Integer>> res = new LinkedList<>();
        Arrays.sort(nums);
        for(int m = 0; m < nums.length-2; m++){
            if(m != 0 && nums[m] == nums[m-1])continue;
            for(int n = m+1; n < nums.length; n++){
                if(n != m+1 && nums[n] == nums[n-1])continue;
                int i = n+1;
                int j = nums.length-1;
                while (i < j){
                    int sum = nums[m] + nums[n] + nums[i] + nums[j];
                    if(sum > target){
                        while (i < j && nums[j] == nums[--j]);
                    }else if(sum < target){
                        while (i < j && nums[i] == nums[++i]);
                    }else {
                        res.add(new LinkedList<>(Arrays.asList(nums[m], nums[n],nums[i],nums[j])));
                        while (i < j && nums[j] == nums[--j]);
                        while (i < j && nums[i] == nums[++i]);
                    }
                }
            }
        }
        return res;
    }

    /**
     * 给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
     * 不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
     * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     */
    public int removeElement(int[] nums, int val){
        Arrays.sort(nums);
        int index = 0;
        for(int i = 0; i < nums.length;i++){
            if(nums[i] != val){
                nums[index++] = nums[i];
            }

        }
        return index;
    }

    /**
     * 实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序列下的一个更大的排列
     * 如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）
     */
    public void nextPermutation(int[] nums){
        int i = nums.length-2;
        while(i >= 0 && nums[i] >= nums[i+1]) i--;
        if(i >= 0){
            int j = nums.length;
            while (j >= 0 && nums[j] <= nums[i]){
                j--;
            }
            swap(nums,i,j);
        }
        reverseSort(nums, i+1);
    }
    public void reverseSort(int[] nums, int i){
        int j = nums.length-1;
        while (i < j){
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
    }
    public void swap(int[] nums,int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    /**
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
     * 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回-1.
     * 算法时间复杂度必须是O（log n）级别。
     */
    public int search(int[] nums, int target){
        int i= 0, j = nums.length-1;
        while (nums[i] > nums[j]){
            if(nums[i] > target){
                while (nums[i] > nums[j]){
                    if(nums[j] <  target)return -1;
                    else if(nums[j] == target)return j;
                    else {
                        j--;
                    }
                }
            }else if(nums[i] < target){i++;}
            else return i;
        }
        while (i < nums.length &&nums[i] <= nums[j]){
            if(nums[i] > target)return -1;
            else if(nums[i] == target)return i;
            else i++;
        }
        return -1;
    }

    /**
     * 给定一个按照升序排列的整数数组nums,和一个目标值targe.
     * 找出给定目标值在数组中的开始位置和结束位置
     */
    public int[] searchRange(int[] nums, int target){
        int i = 0, j = nums.length;
        if(j == 0)return new int[]{-1,-1};
        else if(j == 1)return nums[0] == target?new int[]{0,0}:new int[]{-1,-1};
        j--;
        while (i <= j){
            int mid = (i+j)/2;
            if(nums[mid] > target){
                j = mid - 1;
            }else if(nums[mid] < target){
                i = mid + 1;
            }else {
                j = mid;
                while (mid >=0 && nums[mid] == target)mid--;
                i = mid+1;

                while (j < nums.length && nums[j] == target)j++;
                j--;
                return new int[]{i,j};
            }
        }
        return new int[]{-1,-1};
    }

    /**
     * 给定一个排序树组和一个目标值
     */
    public int searchInsert(int[] nums,int target){
        int left = 0; int right = nums.length-1;
        if(right == -1)return 0;
        else if(right == 0)return nums[0] == target?0:nums[0]>target?0:1;
        if(nums[left] > target)return 0;
        else if(nums[right] < target)return nums.length;
        else {
            while (left <= right){
                int mid = (left + right)/2;
                if(nums[mid] > target)right = mid-1;
                else if(nums[mid] < target)left = mid+1;
                else return mid;
            }
        }
        return (left+right)/2+1;
    }

    /**
     * 给定一个无重复元素的数组candidates 和一个目标target，
     * 找出candidates 中所有可以使数字和为target的组合。
     * candidates  中的数字可以无限制重复被选取。
     *
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target){
        int length = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if(length == 0)return res;
        Arrays.sort(candidates);
        helper(new Stack<>(),res,0,target, candidates);
        return res;
    }

    public void helper(Stack<Integer> stack, List<List<Integer>> res,int index, int residue,int[] candidate){
        if(residue == 0){
            res.add(new LinkedList<>(stack));
        }
        for(int i = index; i < candidate.length && residue-candidate[i] >= 0; i++){
            stack.add(candidate[i]);
            helper(stack, res, i,residue-candidate[i],candidate);
            stack.pop();
        }

    }

    /**
     * 给定一个数组 candidates 和一个目标数 target ，
     * 找出 candidates 中所有可以使数字和为 target 的组合。
     * candidates 中的每个数字在每个组合中只能使用一次。
     */
    public List<List<Integer>> combinationSum2(int[] candidates,int target){
        List<List<Integer>> res = new LinkedList<>();
        int length = candidates.length;
        if(length == 0)return res;
        Arrays.sort(candidates);
        findcombinationSum2(res, new Stack<>(), candidates, 0, target);
        return res;

    }
    public void findcombinationSum2(List<List<Integer>> res, Stack<Integer> stack, int[] candidates, int index, int residue){
        if(residue == 0){
            res.add(new LinkedList<>(stack));
        }
        for(int i = index; i < candidates.length && residue-candidates[i] >= 0;i++){
            if(i > index && candidates[i] == candidates[i-1])continue;
            stack.push(candidates[i]);
            findcombinationSum2(res,stack,candidates,i+1,residue-candidates[i]);
            stack.pop();
        }
    }

    /**
     * 给定一个 n*x 的二维矩阵表示一个图像。
     * 将图像顺时针旋转90度。
     *
     * */
    public void rotate(int[][] matrix){
        int len = matrix.length;
        for(int i = 0; i < len; i++){
            for(int j = i; j< len; j++){
                int tem = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tem;
            }
        }
        for(int i = 0; i < len; i++){
            int m = 0, n = len-1;
            while (m < n){
                int tem = matrix[i][m];
                matrix[i][m] = matrix[i][n];
                matrix[i][n] = tem;
                m++;
                n--;
            }
        }
    }
    public void rotate1(int[][] matrix){
        int n = matrix.length;
        for(int i = 0; i < n/2+n%2; i ++){
            for(int j = 0; j < n/2; j++){
                int[] tmp = new int[4];
                int row = i;
                int col = j;
                for(int k = 0; k < 4; k++){
                    tmp[k] = matrix[row][col];
                    int x = row;
                    row = col;
                    col = n-1-x;
                }
                for(int k = 0; k < 4; k++){
                    matrix[row][col] = tmp[(k+3)%4];
                    int x = row;
                    row = col;
                    col = n - 1 - x;
                }
            }
        }
    }
    // 给定一个包含m*n个元素的矩阵，请按照顺时针螺旋顺序，返回矩阵中的所有元素
    public List<Integer> spiralOrder(int[][] matrix){
        List<Integer> list = new LinkedList<>();
        int rowLen = matrix.length;
        if(rowLen == 0)return list;
        int colLen = matrix[0].length;
        int rowStart = 0, colStrat = 0;
        while (rowStart < rowLen && colStrat < colLen){
            int i = rowStart, j = colStrat;
            while (j < colLen){
                list.add(matrix[i][j]);
                j++;
            }
            rowStart++;
            while (++i < rowLen){
                list.add(matrix[i][j-1]);
            }

            j = --colLen;
            while (--j >=colStrat && rowStart < rowLen){
                list.add(matrix[i-1][j]);
            }
            i = --rowLen;
            while (--i >= rowStart && colStrat < colLen){
                list.add(matrix[i][j+1]);
            }
            colStrat++;
        }
        return list;
    }

    /**
     * 给定一个非负整数数组，你最初位于数组的第一个位置
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 判断你是否能够到达最后一个位置。
     */
    public boolean canJumpFromPosition(int position, int[] nums) {
        if (position == nums.length - 1) {
            return true;
        }

        int furthestJump = Math.min(position + nums[position], nums.length - 1);
        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
            if (canJumpFromPosition(nextPosition, nums)) {
                return true;
            }
        }

        return false;
    }

    public boolean canJump(int[] nums) {
        return canJumpFromPosition(0, nums);
    }

    Index[] memo;
    public boolean isCanJumpLast(int position, int[] nums){
        if(memo[position] != Index.UNKNOWN){
            return memo[position]==Index.GOOD;
        }
        int furthestJump = Math.min(position+nums[position], nums.length-1);
        for(int nextPosition = furthestJump; nextPosition > position; nextPosition--){
            if(isCanJumpLast(nextPosition,nums)){
                memo[position] = Index.GOOD;
                return true;
            }
        }
        memo[position] = Index.BAD;
        return false;
    }
    public boolean canJump1(int[] nums){
        memo = new Index[nums.length];
        for(int i = 0; i < nums.length; i++){
            memo[i] = Index.UNKNOWN;
        }
        memo[memo.length-1] = Index.GOOD;
        return isCanJumpLast(0,nums);
    }

    public boolean canJump2(int[] nums){
        Index[] memo = new Index[nums.length];
        for(int i = 0; i < memo.length; i++){
            memo[i] = Index.UNKNOWN;
        }
        memo[memo.length-1] = Index.GOOD;
        for(int i = nums.length-2; i >= 0; i--){
            int furthestJump = Math.min(i+nums[i], nums.length-1);
            for(int j = i+1; j <= furthestJump ;j++){
                if(memo[j] == Index.GOOD){
                    memo[i] = Index.GOOD;
                    break;
                }
            }
        }
        return memo[0] == Index.GOOD;
    }

    //贪心算法
    public boolean canJump3(int[] nums){
        int last_pos = nums.length-1;
        for(int i = nums.length-1; i >= 0; i--){
            if(i+nums[i] >= last_pos){
                last_pos = i;
            }

        }
        return last_pos == 0;
    }

    /**
     * 给出一个区间的集合，请合并所有重叠的区间
     *
     */
    public int[][] merge(int[][] intervals){
        List<int []> list = new ArrayList<>();
        if(intervals.length == 0)return list.toArray(new int[0][]);
        Arrays.sort(intervals,(a,b) -> a[0] -b[0]);
        int i = 0;
        while(i < intervals.length){
            int left = intervals[i][0];
            int right = intervals[i][1];
            while (i < intervals.length-1 && intervals[i+1][0] <= right){
                i++;
                right = Math.max(right, intervals[i][1]);
            }
            list.add(new int[]{left, right});
            i++;
        }
        return list.toArray(new int[list.size()][]);
    }

    //给定一个正整数n,生成一个包含1到n2所有元素，且元素按顺时针螺旋排列的正方形矩阵。
    public int[][] generateMatrix(int n){
        int[][] arr = new int[n][n];
        int rowStart = 0, colStart = 0;
        int rowLen = n,colLen = n;
        int m = 1;
        while (rowStart < n && colStart < n){
            int i = rowStart, j = colStart;
            while (j < colLen){
                arr[i][j] = m++;
                j++;
            }
            rowStart++;
            while (++i < rowLen){
                arr[i][j-1] = m++;
            }
            j = --colLen;
            while (--j >= colStart){
                arr[i-1][j] = m++;
            }
            i = --rowLen;
            while (--i >= rowStart){
                arr[i][j+1] = m++;
            }
            colStart++;
        }
        return arr;
    }

    public int uniquePaths(int m, int n){
        int[][] dp = new int[m][n];
        for(int i = 0; i < m; i++){
            dp[0][i] = 1;
        }
        for(int j = 0; j < n; j++){
            dp[j][0] = 1;
        }
        for(int i = 1; i < m;i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i-1][j]+ dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
    static int[][] a = new int[101][101];

    public int uniquePaths1(int m, int n){
        if(m <= 0 || n <= 0){
            return 0;
        }
        if(m ==1 || n == 1)return 1;
        else if(m == 2 && n == 2)
            return 2;
        else if((m == 3 && n == 2) || (m == 2 && n == 3))
            return 3;
        if(a[m][n] > 0)return a[m][n];
        a[m-1][n] = uniquePaths(m-1,n);
        a[m][n-1] = uniquePaths(m,n-1);
        a[m][n] = a[m-1][n] + a[m][n-1];
        return a[m][n];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid){
        if(obstacleGrid[0][0] == 1)return 0;
        obstacleGrid[0][0] = 1;
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        boolean mark = true;
        for(int i = 1; i < n; i++){
            if(mark && obstacleGrid[0][i] == 0) obstacleGrid[0][i] = 1;
            else {
                obstacleGrid[0][i] = 0;
                mark = false;
            }
        }
        mark = true;
        for(int i = 1; i < m; i++){
            if(mark && obstacleGrid[i][0] == 0) obstacleGrid[i][0] = 1;
            else {
                obstacleGrid[i][0] = 0;
                mark = false;
            }
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                if(obstacleGrid[i][j] == 0){
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1];
                }else {
                    obstacleGrid[i][j] = 0;
                }
            }
        }
        return obstacleGrid[m-1][n-1];
    }

    /**
     * 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     */
    public int minPathSum(int[][] grid){
        int r = grid.length;
        int c = grid[0].length;
        for(int i = 1; i < r;i++){
            grid[i][0] += grid[i-1][0];
        }
        for(int i = 1; i < c; i++){
            grid[0][i] += grid[0][i-1];
        }
        for(int i = 1; i < r; i++){
            for(int j = 1; j < c; j++){
                grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[r-1][c-1];
    }

    public int[] plusOne(int[] digits){
        for(int i = digits.length-1; i >= 0; i--){
            digits[i]++;
            digits[i] %= 10;
            if(digits[i] != 0)return digits;
        }
        digits = new int[digits.length+1];
        digits[0] = 1;
        return digits;
    }

    public void setZeros(int[][] matrix){
        boolean isCol = false;
        int R = matrix.length;
        int C = matrix[0].length;
        for(int i = 0;i < R; i++){
            if(matrix[i][0] == 0) isCol = true;
            for(int j = 1; j < C; j++){
                if(matrix[i][j] == 0){
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for(int i = 1; i < R; i++) {
            for(int j = 1; j < C; j++){
                if(matrix[i][0] == 0 || matrix[0][j] == 0){
                    matrix[i][j] = 0;
                }
            }
        }
        //判断第一行是否需要置零
        if(matrix[0][0] == 0)
            for(int i = 1; i < C; i++){
                matrix[0][i] = 0;
            }
        if(isCol)
            for(int i =0; i < R; i++){
                matrix[i][0] = 0;
            }
    }
    public static void main(String[] args){
        int[][] matrix= new int[][]{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
        int[] height = {1,8,6,2,5,4,8,3,7};
        int[] nums = {3,2,1,0,4};
        int target = 2;
        int[][] intervals = new int[][]{{1,3},{2,6},{8,10},{15,18}};
        Solution so = new Solution();
        int a = so.uniquePaths1(7,3);
        System.out.println(a);
        //List<Integer> lis = so.spiralOrder(matrix);
        // int a = so.searchInsert(nums,target);
        //List<List<Integer>> a = so.fourSum(nums,target);
        //System.out.println(a);
    }
}


enum Index{
    GOOD, BAD, UNKNOWN
}