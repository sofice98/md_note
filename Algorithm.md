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
int btree[1<<maxd];//最大节点个数为2^maxd-1,编号为[1,2^maxd-1]
btree[2k];btree[2k+1];//左右子树
//链式存储
typedef struct Node {
    int value;
    Node *parent, *left, *right;
    Node(int v = 0) :value(v), parent(nullptr), left(nullptr), right(nullptr) {}
}BTree;
//有根树
//左孩子右兄弟
typedef struct Node{ 
    int value;
    Node *parent, *left, *right;
    Node(int v = 0) :value(v), parent(nullptr), left(nullptr), right(nullptr) {}
}Tree;
//孩子指针数组
typedef struct node {
    int value;
    vector<node*> children;
};
```



## 图

```c++
//邻接矩阵
//获取uv关系，添加删除边都为o(1)
//稀疏图浪费空间；基本型只能记录uv间一个关系
int graph[maxv][maxv];

//邻接表
//只需o(E)空间
//获取uv关系时非o(1)；难以有效删除边
vector<vector<int>> graph;
vector<int> graph[MAX];
//加权邻接表，first为终点，second为边权值
vector<pair<int, int>> graph[MAX];

//边结点
struct Edge {
    int u, v, w;
    Edge(int u = 0,int v = 0,int w = 0) :
        u(u), v(v), w(w) {}
    bool operator< (const Edge& e) const {
        return w < e.w;
    }
};    
```

**图遍历**

```c++
int G[maxv][maxv];
bool vis[maxv];
int n;
int u, v;
f(i, 1, m) {
    scanf("%d %d", &u, &v);
    G[u][v] = 1; G[v][u] = 1;
}
//bfs
void bfs(int cur) {
    queue<int> q;
    q.push(cur);
    while (!q.empty()) {
        int i = q.front();
        q.pop();
        vis[i] = true;
        for (int j = 0; j < n; j++) {
            if (!vis[j] && G[i][j])    q.push(j);
        }
    }

}
void bfsTraverse() {
    memset(vis, 0, sizeof(vis));
    for (int i = 0; i < n; i++) {
        if (!vis[i])     bfs(i);
    }
}
//dfs
void dfs(int u) {
    vis[u] = true;
    for (int v = 0; v < n; v++) {
        if (!vis[v] && G[u][v])    dfs(v);
    }
}
void dfsTraverse() {
    memset(vis, 0, sizeof(vis));
    for (int i = 0; i < n; i++) {
        if (!vis[i])     dfs(i);
    }
}
```

**使用数组来变换方向**

```c++
const int dx[] = { -1,1,0,0 };  
const int dy[] = { 0,0,-1,1 };
int newx = x + dx[d];
int newy = y + dy[d];
```

**最短路径**

```c++
int n;
vector<pair<int, int>> graph[MAX];//加权邻接表
int dist[MAX];//距离表
bool vis[MAX];//访问表
int f[MAX];//父节点

//迪杰斯特拉算法+优先级队列，Tn=o((V+E)logV)
void Dijkstra(int src) {
    //初始化
    fill(dist, dist + MAX, INF);
    memset(vis, 0, sizeof(vis));
    dist[src] = 0; parent[src] = -1;
    //优先级队列，pair<dist[i],i>，dist小的优先
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push(make_pair(0, src)); 
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        if (vis[u]) continue;
        vis[u] = true;
        for (auto& item : graph[u]) {
            int v = item.first;
            int w = item.second;
            if (!vis[v] && dist[v] > d + w) {
                dist[v] = d + w;
                pq.push(make_pair(dist[v], v));
                parent[v] = u;
            }
        }
    }
}
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



## 并查集

一种用互质集合对数据进行分类管理的数据结构

主要包括合并与查找操作

```c++
struct DisjointSet {
    vector<int> rank, p;//高度，父指针
    DisjointSet() {}
    DisjointSet(int size) {
        rank.resize(size, 0);
        p.resize(size, 0);
        for (int i = 0; i < size; i++)     makeSet(i);
    }
    //建立
    void makeSet(int x) {
        p[x] = x;
        rank[x] = 0;
    }
    //合并
    void unite(int x, int y) {
        int xp = findSet(x);
        int yp = findSet(y);
        if (rank[xp] > rank[yp])  p[yp] = xp;
        else {
            p[xp] = yp;
            if (rank[xp] == rank[yp])    rank[yp]++;
        }
    }
    //查找
    int findSet(int x) {
        if (x != p[x]) {
            p[x] = findSet(p[x]);
        }
        return p[x];
    }

    bool same(int x, int y) {
        return findSet(x) == findSet(y);
    }
};
```



## KDTree

用于k维的范围搜索树

```c++
struct Node {//树节点
    int location;
    int p, l, r;
    Node() {}
};
struct Point {//数据点
    int id, x, y;
    Point() {}
    Point(int id, int x, int y) :id(id), x(x), y(y) {}
    bool operator<(const Point& p) const {
        return id < p.id;
    }
    void print() {
        printf("%d\n", id);
    }
};
//建树，Tn=o(nlog^2n)
int makeKDTree(int l, int r, int depth) {
    if (l >= r) return NIL;
    int mid = l + r >> 1;
    int t = np++;
    if (depth % 2 == 0)  sort(P + l, P + r, lessX);
    else    sort(P + l, P + r, lessY);
    T[t].location = mid;
    T[t].l = makeKDTree(l, mid, depth + 1);
    T[t].r = makeKDTree(mid + 1, r, depth + 1);
    return t;
}
//查找，Tn=o(n^(1-1/k)+d)，d为指定范围内点的数量
void find(int v, int sx, int tx, int sy, int ty, int depth, vector<Point>& ans) {
    int x = P[T[v].location].x;
    int y = P[T[v].location].y;

    if (sx <= x && x <= tx && sy <= y && y <= ty)
        ans.push_back(P[T[v].location]);
    if (depth % 2 == 0) {
        if (T[v].l != NIL && sx<=x) 
            find(T[v].l, sx, tx, sy, ty, depth + 1, ans);
        if (T[v].r != NIL && x <= tx)
            find(T[v].r, sx, tx, sy, ty, depth + 1, ans);
    }
    else {
        if (T[v].l != NIL && sy <= y)
            find(T[v].l, sx, tx, sy, ty, depth + 1, ans);
        if (T[v].r != NIL && y <= ty)
            find(T[v].r, sx, tx, sy, ty, depth + 1, ans);
    }
}
```



## 线段树



## 树状数组



## 单调队列

求数组中一段滑动长度内的最大值或最小值，Tn=o(n)

```c++
typedef struct node {        //队列的节点，包含元素在列表中原来的位置和值
    int order;
    int value;
};
deque<node> hq;    //定义节点类型单调队列
vector<int> m;      //用于储存最大值序列

int n, k, t;           //滑动窗口长度为k，t用于暂时储存输入
node tmp;
scanf("%d%d", &n, &k);
for (int i = 0; i < n; i++) {
    scanf("%d", &t);
    while (!hq.empty() && i - hq.front().order >= k) hq.pop_front(); //剔除队头过期元素   
    while (!hq.empty() && hq.back().value <= t) hq.pop_back();     //剔除队尾小于将入列的值，保证队头为最大值
    tmp.value = t;  //节点入列
    tmp.order = i;
    hq.push_back(tmp);
    if (i >= k - 1) //开始输出           
        m.push_back(hq.front().value);
}
//前缀和+单调队列
//求区间长度[s,t]的最大子串和
int a[MAX];//[1...n]
int sum[MAX];//前缀和[1...n]
sum[0] = 0;//考虑从第一个元素开始
for (int i = 1; i <= n; i++)	sum[i] = sum[i - 1] + a[i];
hq.push_back(0);
for (int i = 1; i <= n; i++) {
    while (!hq.empty() && sum[hq.back()] > sum[i]) hq.pop_back();
    hq.push_back(i);
    while (!hq.empty() && t < i - hq.front()) hq.pop_front();
    ans = max(ans, sum[i] - sum[hq.front()]);
}
printf("%d\n", ans);
```





# 搜索

## 二分搜索

满足单调，有最大最小值

一般用于求最小化最大值、最大化最小值

```c++
//二分模版
int left = 0;
int right = n;
while (left < right) {
    int mid = left + right >> 1;
    if (A[mid] == key) return mid;
    else if (key < A[mid]) right = mid;
    else    left = mid + 1;
}
//双分支，解决问题常用，关键在于建模和check()   
int ans;          //记录答案
while (right - left > 1) {//一定要有1，否则跳不出循环
        int mid = left + right >> 1;
        if (check(mid)){  //检查条件，如果成立
            ans = mid;    
            left = mid;
        }
        else   right = mid;        
}
//实数二分
while(right - left > eps)  　{ ... } //给定精度
for(int i = 0; i < 100; i++) { ... }//精度为1/2^100

//三分法求单峰、谷极值 
//实数
while(R-L > eps){  
    double k =(R-L)/3.0;
    double mid1 = L+k, mid2 = R-k;
    if(check(mid1) > check(mid2)) 
        R = mid2;
    else   L = mid1;
}
//整数
while(R - L > 1){  
    int mid1 = left + (right - left)/3;
    int mid2 = right- (right - left)/3;
    if(check(mid1) > check(mid2))
        R = mid2;
    else   L = mid1;
}
```

lower_bound(起始地址，结束地址，要查找的数值) 返回的是数值 **第一个** 出现的位置，大于等于

upper_bound(起始地址，结束地址，要查找的数值) 返回的是数值 **最后一个** 出现的位置，大于

binary_search(起始地址，结束地址，要查找的数值)  返回的是是否存在这么一个数，是一个**bool值**

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

**IDA***

当设计出乐观估计函数，预测从当前节点至少还需要扩展几层节点才有可能得到解，则变成了IDA*











# 动态规划

具有最优子结构（全局最优解包含局部最优解），可以用状态转移（状态转移方程）解决的一个方法

直接递归时，效率往往底下，原因是相同的子问题被重复计算了多次

设置dp含义，求dp转移方程，设置剪枝

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

当使用的数据为当前行和正上方一个时（dp[i]\[..],dp[i-1]\[j]），可以将dp数组压缩成一维数组：

```c++
int dp[MAX][MAX];//使用前i枚硬币支付j元时所需最少枚数
int dpzip[MAX];//压缩dp
int coins[MAX];
int m, n;//m种硬币，支付n元
//打表dp
int solve() {
    for (int j = 1; j <= n; j++) dp[0][j] = INT_MAX;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (j - coins[i] >= 0)
                dp[i][j] = min(dp[i - 1][j], dp[i][j - coins[i]] + 1);
            else
                dp[i][j] = dp[i - 1][j];
        }
    }
    return dp[m][n];
}
//压缩dp矩阵至一维，减少不需要的空间,也会减少约一半时间，缺点是不能找到使用了哪几个面值的硬币
int solve2() {
    for (int j = 1; j <= n; j++) dpzip[j] = INT_MAX;
    for (int i = 1; i <= m; i++) {
        for (int j = coins[i]; j <= n; j++) {
            dpzip[j] = min(dpzip[j], dpzip[j - coins[i]] + 1);
        }
    }
    return dpzip[n];
}
```





**DAG(Directed acyclic graph)**

许多动态规划问题都可以转化为DAG上的最长路最短路或路径计数问题





# 字符串

子串要连续，子序列不要求连续



# 数论

## 最大公因数

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

## 全排列映射到整数

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





# 排序

常见排序算法：

| 排序算法          | 最好T(n) | 平均T(n) | 最坏T(n) | S(n) | 稳定性 | 每趟全局有序 | 链表         |
| ----------------- | -------- | -------- | -------- | ---- | ------ | ------------ | ------------ |
| 直接插入/折半插入 | n        | n2       | n2       | 1    | 1      | 0            | 单/0         |
| 希尔排序          | n        | n1.25    | n1.5     | 1    | 0      | 0            | 0            |
| 冒泡排序          | n        | n2       | n2       | 1    | 1      | 1            | 顺链表顺序冒 |
| 快速排序          | nlogn    | nlogn    | n2       | logn | 0      | 1            | 单           |
| 简单选择排序      | n2       | n2       | n2       | 1    | 0      | 1            | 单           |
| 堆排序            | nlogn    | nlogn    | nlogn    | 1    | 0      | 1            | 树链         |
| 归并排序          | nlogn    | nlogn    | nlogn    | n    | 1      | 0            | 0            |
| 基数排序          | d(n+r)   | d(n+r)   | d(n+r)   | r    | 1      | 0            | 单           |
| 计数排序          | n+k      | n+k      | n+k      | k    | 1      | 1            | 单           |



# STL

**vector动态数组**：insert(p,x)，erase(p)都为o(n)

**stack栈**：所有操作o(1)

**queue队列**：所有操作o(1)

**list双向链表**：可以在front处操作，insert(p,x)，erase(p)都为o(1)

**set集合**：insert(key)，erase(key)，find(key)都为o(logn)

**map字典**：insert( (key，val) )，erase(key)，find(key)都为o(logn)

**priority_queue优先级队列**：默认大根堆

```c++
//降序队列，大顶堆，大的优先
priority_queue <int,vector<int>,less<int> >q;
//升序队列，小顶堆，小的优先，或者默认的乘-1
priority_queue <int,vector<int>,greater<int> > q;
//自定义排序，重载仿函数
struct cmp{
    bool operator()(const pair<int, int> &a, const pair<int, int> &b){
            return a.first + a.second < b.first + b.second;//返回true时，说明a的优先级低于b
    }
};
priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> pq;
//自定义node，重载操作符
struct node{
      int x, y;
      node(int x, int y):x(x),y(y){}
      bool operator< (const node &b) const {
           if(x == b.x)  return y >= b.y;
           else return x > b.x;
      }
};
priority_queue<node> pq;
```



# 小技巧

+ 常用变量

  ```c++
  #include<bits/stdc++.h>
  using namespace std;
  typedef long long ll;
  constexpr int MOD = 1e9 + 7;
  constexpr int INF = 0x7fffffff;
  constexpr int MAX = 10000;
  constexpr double eps = 1e-7;
  #define f(i, a, b) for(int i = a;i <= b;i++)
  #define equals(a, b) ( fabs( (a) - (b) ) < eps )
  ```

+ 编码解码映射哈希可提高速度

+ 提前求出素数表，阶乘表或多个test公共变量，可加快速度

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
  
+ 枚举排列：

  ```c++
void print_permutation(序列A，集合S){
      if(S空) 输出序列A
    else 按照顺序考虑S的每个元素v{
          print_permutation(在A末尾添加v，S-{v})
      }       
  }
  ```
  
  或使用next_permutation函数
  
+ 生成一个大随机数
  ```c++
  unsigned long ulrand(){          
      return (
        (((unsigned long)rand()<<24)& 0xFF000000ul)
       |(((unsigned long)rand()<<12)& 0x00FFF000ul)
       |(((unsigned long)rand())    & 0x00000FFFul));
  }
  ```
  
+ 快速读
  
  ```c++
  iostream::sync_with_stdio(false);
  inline int read() {
        int ret = 0, op = 1;
        char c = getchar();
        while (!isdigit(c)) {
            if (c == '-') op = -1; 
            c = getchar();
        }
        while (isdigit(c)) {
            ret = (ret << 3) + (ret << 1) + c - '0';
            c = getchar();
	      }
	      return ret * op;
	  }
	```





# 注意事项

+ 对每个int变量关注其值是否会超过int范围，改为long或long long
+ 赋值时记得类型转换
+ 一定要加上对特殊情况的判别，不但会提高正确率，还能减少Runtime（如开头加上对空参数的判别输出）
+ 大数组申请为全局变量
+ 即使是暴力枚举，也是要认真分析问题的，以减少枚举量



















