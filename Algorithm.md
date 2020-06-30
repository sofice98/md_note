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



