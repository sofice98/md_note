[toc]

# 数据结构

## 链表

```c++
//顺序存储
char s[maxn];
int next[maxn];
//链式存储
typedef struct Node {
    int value;
    Node *next;
    Node(int v = 0) :value(v), next(nullptr) {}
}Linklist;
//插入
Linklist* head = new Node();
Node* p = head;
int n;
while (scanf("%d", &n) == 1) {
    Node* newnode = new Node(n);
    p->next = newnode;
    p = p->next;
}
```



## 树

```c++
//二叉树
//顺序存储
const int maxd = 20//最大深度
int tree[1<<maxd];//最大节点个数为2^maxd-1,编号为[1,2^maxd-1]
tree[2k];tree[2k+1];//左右子树
//链式存储
typedef struct Node {
    bool flag;
    int value;
    Node *left, *right;
    Node(int v = 0) :flag(false), value(v), left(nullptr), right(nullptr) {}
}Tree;
```



## 图

```c++
//顺序存储
int G[max][max];//邻接矩阵
vector<vector<int>> G;//邻接表
//链式存储
```

**使用数组来变换方向**

```c++
const int dx[] = { -1,1,0,0 };  
const int dy[] = { 0,0,-1,1 };
int newx = x + dx[d];
int newy = y + dy[d];
```

**拓扑排序**



**欧拉回路**

除了起点和终点，其他点的度数应为偶数（进出次数相同），并且图连通。度数为奇数的点最多两个并且为起点和终点

```c++
void euler(int u){
    for(int v =0; v < n; v++)
        if(G[u][v] && !vis[u][v]){
            vis[u][v] = vis[v][u] = 1;
            euler(v);
            printf("%d %d\n", u, v);
        }
}
```





# 暴力法

## 枚举排列

即使是暴力枚举，也是要认真分析问题的，以减少枚举量

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

递归函数不满足条件时，不再调用，而是返回上一层调用（递归枚举）

当修改全局变量时，在出口处应当改回来

```c++
void dfs(int cur) {
    if (cur == n) {//到达规定最深处
        //输出结果
        printf("%d", res);
        return;
    }
    //不满足则不继续递归，剪枝
    if (flag) {
        res[cur] = i;//赋值
        vis[i] = true;//设置访问标志
        dfs(cur + 1);//递归搜索
        vis[i] = false;//解除访问标志
    }
    
}
```

> Eight Queens，UVA524

## 状态空间搜索

找到一个从初始状态到终止状态的路径，等价于隐式图的最短路径查找

```c++
typedef int State[9];           //定义状态类型
const int maxstate = 1000000;	//最大状态数
State st[maxstate], goal;       //状态数组
int dist[maxstate];             //距离数组
int vis[362880], fact[9];       //状态访问数组和阶乘表
int fa[maxstate];               //父节点编号
const int dx[] = { -1,1,0,0 };	//用数组移动
const int dy[] = { 0,0,-1,1 };
int front = 1, rear = 2;		//头尾指针

int bfs() {
    init_lookup_table();//初始化查找表
    while (front < rear) {
        State& s = st[front];//当前状态
        if(memcmp(goal, s, sizeof(s)) == 0)    return front;//找到目标状态
        for (int d = 0; d < 4; d++) {//尝试搜索下一个相邻状态
            if (flag) {//如果合法
                State& t = st[rear];//扩展新节点
                memcpy(&t, &s, sizeof(s));
                dist[rear] = dist[front] + 1;//更新距离值
                if (try_to_insert(rear)) {
                    fa[rear] = front;
                    rear++;
                }
            }
        }
        front++;
    }
    return 0;//失败
}
```

需要输出过程时，要加上父指针和状态数组

> Eight digital

## 迭代加深搜索(IDDFS)

在DFS的搜索里面，可能会面临一个答案的层数很低，但是DFS搜到了另为一个层数很深的分支里面导致时间很慢，但是又卡BFS的空间

设置递归深度上限的回溯法，每轮依次增加最大递归深度，在深度没有明显上限以及宽度太大的时候，可以使用这种方式

+ 要求必须有解
+ 时间复杂度只比BFS稍差一点（虽然搜索k+1层时会重复搜索k层，但是整体而言并不比广搜慢很多）
+ 空间复杂度与深搜相同，却比广搜小很多

```c++
bool dfs(int cur) {
    if (cur == maxd) {
        if (flag) return false;//没找到解
        if (better(cur))   memcpy(ans, v, sizeof(LL) * (d + 1));//当前解更优
        return true;
    }
    bool ok = false;
    for (int i = from;; i++) {//遍历所有后继状态
        if (bb * (maxd + 1 - d) <= i * aa) break;//剪枝
        v[d] = i;//赋值
        if (dfs(cur + 1))  ok = true;
    }
    return ok;
}

for (maxd = 1;; maxd++) {//迭代加深，每一次循环增加递归最大深度
    memset(ans, -1, sizeof(ans));//初始化结果    
    if (dfs(0)) {//如果找到结果
    	ok = 1; break;//退出循环
    }
}
```

**IDA**

当设计出乐观估计函数，预测从当前节点至少还需要扩展几层节点才有可能得到解，则变成了IDA*



> UVA11212，UVA12558



# 动态规划

具有最优子结构（全局最优解包含局部最优解），可以用状态转移（状态转移方程）解决的一个方法

直接递归时，效率往往底下，原因是相同的子问题被重复计算了多次

```c++
//数字三角形
//递推计算法
for (int j = 1; j < n; j++)    dp[n][j] = a[n][j];//计算最后一列
for (int i = n - 1; i >= 1; i--)//从下往上递推
    for (int j = 1; j <= i; j++)
        dp[i][j] = a[i][j] + max(dp[i + 1][j], dp[i + 1][j + 1]);
//记忆化搜索
memset(dp, -1, sizeof(d));//如果为-1则未访问过
int solve(int i, int j) {
    if (dp[i][j] >= 0)  return dp[i][j];//直接取之前计算过的
    return dp[i][j] = a[i][j] + (i == n ? 0 : max(solve(i + 1, j), solve(i + 1, j + 1)));
}
```



**DAG(Directed acyclic graph)**

许多动态规划问题都可以转化为DAG上的最长路最短路或路径计数问题





# 数论

```c++
//辗转相除法，递归开销大，要求a>=b
int gcd(LL a, LL b) {
    return a % b == 0 ? b : gcd(b, a % b);
}
//更相减损法
int qGCD(int a, int b)
{
	if(a == 0) return b;
	if(b == 0) return a;
	if(!(a & 1) && !(b & 1)) // a % 2 == 0 && b % 2 == 0;
		return qGCD(a >> 1, b >> 1) << 1;
	else if(!(b & 1))
		return qGCD(a, b >> 1);
	else if(!(a & 1))
		return qGCD(a >> 1, b);
	else
		return qGCD(abs(a - b), min(a, b));
}
```



# 编码

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

+ 编码解码映射哈希可提高速度

+ 提前求出素数表，阶乘表可加快速度

+ 一定要加上对特殊情况的判别，不但会提高正确率，还能减少Runtime（如开头加上对空参数的判别输出）

+ ```c++
  while (scanf("%d", &n) == 1 && n != 0)//n=0结束
  while (scanf("%d", &n) != EOF)//到文档末尾结束
  ```

+  ```c++
  //重定向文件IO
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  //不使用重定向的文件IO
  FILE* fin, * fout;
  fin = fopen("data.in", "rb");//标准fin=stdin,fout=stdout
  fout = fopen("data.out", "wb");
  fscanf(fin, "%d", &x);
  fprintf(fout, "%d", x);
  ```
  
+ 





















