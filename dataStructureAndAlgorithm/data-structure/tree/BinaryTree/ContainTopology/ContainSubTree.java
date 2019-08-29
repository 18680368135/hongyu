package tree.BinaryTree.ContainTopology;

public class ContainSubTree {


    public boolean isSubTree(Node t1, Node t2){
        String t1Str = serialByPre(t1);
        String t2Str = serialByPre(t2);
        return getIndexOf(t1Str, t2Str) != -1;

    }

    public String serialByPre(Node head){
        if(head == null) return "#!";
        String res = head.value + "!";

        res += serialByPre(head.left);
        res += serialByPre(head.right);
        return res;
    }

    //KMP
    public int getIndexOf(String s, String m){
        if(s == null || m == null || s.length() < 1 || s.length() < m.length()) return -1;

        char[] ss = s.toCharArray();
        char[] ms = m.toCharArray();

        int si = 0;
        int mi = 0;

        int[] next = getNextArray(ms);

        while(si < ss.length && mi < ms.length){
            if(ss[si] == ms[mi]){
                si++;
                mi++;
            }else if(next[mi] == -1){
                si++;
            }else{
//                si++;
                mi = next[mi];
            }

        }
        si = mi == ms.length ? si-mi : -1;
        return si;

    }
    public int[] getNextArray(char[] ms){
        if(ms.length == 1) return new int[] {-1};

        int[] next = new int[ms.length];

        next[0] = -1;
        next[1] = 0;

        int pos = 2;
        int cn = 0;

        while(pos < ms.length){
            if(ms[pos-1] == ms[cn]){
                next[pos++] = ++cn;
            }else if(cn > 0){
                cn = next[cn];
            }else{
                next[pos++] = 0;
            }
        }

        return next;
    }
}
