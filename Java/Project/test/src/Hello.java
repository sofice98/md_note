import java.lang.reflect.*;
import java.lang.Integer;
import java.util.*;

public class Hello {
    public static void main(String[] args) throws Exception   {
        // 创建 Map 集合对象
        HashMap<Integer, String> map = new HashMap<Integer,String>();
        // 添加元素到集合
        map.put(1, "小五");
        map.put(2, "小红");
        map.put(3, "小张");

        // 获取所有的键  获取键集
        Set<Integer> keys = map.keySet();
        // 遍历键集 得到 每一个键
        for (Integer key : keys) {
            // 获取对应值
            String value = map.get(key);
            System.out.println(key + "：" + value);
        }
        ArrayList<Integer> a = new ArrayList<Integer>();

    }
}

