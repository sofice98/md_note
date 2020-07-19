/**********
*DP
*Sofice
*2020/7/10
*************/
#include<bits/stdc++.h>
using namespace std;
#define MAX_N 101
int n, W;
int w[MAX_N], v[MAX_N];//[1...n]
int dp[MAX_N][MAX_N];//最优子v值

int solve(){
    memset(dp, 0, sizeof(dp));
    //从第1个到第i个物品选出最优的不超过j的解
    for (int i = 1; i<= n; i++) {
        for (int j = 0; j <= W; j++) {
            if (j < w[i])  dp[i][j] = dp[i - 1][j];//已经不超过了
            else           dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);//分别为不选和选
        }
    }
    return dp[n][W];
}

int main() {
    memset(w, 0, sizeof(w));
    memset(v, 0, sizeof(v));
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) 
        scanf("%d %d", &w[i], &v[i]);
    scanf("%d", &W);
    printf("%d", solve());
    


    system("pause");
    return 0;
}