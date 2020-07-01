# 暴力法

## 枚举排列

```c++
void print_permutation(序列A，集合S){
    if(S空) 输出序列A
    else 按照顺序考虑S的每个元素v{
        print_permutation(在A末尾添加v，S-{v})
    }       
}
```

或使用next_permutation函数

## 子集生成

二进制法：一位表示一个元素是否在集合中

```c++
void print_subset(int n, int s) {//打印整数s表示的子集
    for (int i = 0; i < n; i++)
        if (s & (1 << i)) printf("%d ", i);
    printf("\n");
}
void generate_subset(int n) {
    //枚举整数
    for (int i = 1; i < (1 << n); i++) {
        print_subset(n, i);
    }
}
```



## 回溯法

递归函数不满足条件时，不再调用，而是返回上一层调用

当修改全局变量时，在出口处应当改回来



## 状态空间搜索

找到一个从初始状态到终止状态的路径



**将全排列映射到整数：**

```c++
int vis[362880], fact[9];       //状态访问数组和阶乘表
void init_lookup_table() {
    fact[0] = 1;
    for (int i = 1; i < 9; i++)    fact[i] = fact[i - 1] * i;
}

int try_to_insert(int s) {
    int code = 0;//把st[s]映射到整数code
    for (int i = 0; i < 9; i++) {
        int cnt = 0;
        for (int j = i + 1; j < 9; j++)  if (st[s][j] < st[s][i])   cnt++;
        code += fact[8 - i] * cnt;
    }
    if (vis[code])   return 0;
    return vis[code] = 1;
}
```





# 技巧

+ 使用数组来变换方向
+ 编码解码映射哈希可提高速度
+ 提前求出素数表，阶乘表可加快速度





















