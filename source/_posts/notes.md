---
title: notes
date: 2024-03-15 15:14:22
categories:
    - 算法
tags:
cover:
    /img/cover6.jpg
top_img:
    /img/top_img.jpg
---

# [字典树（前缀树）求区间异或和（异或对）最大值](https://blog.51cto.com/u_15064643/3333274)

## [求子区间异或对最大值](https://ac.nowcoder.com/acm/contest/11171/B)

求子区间异或对的最大值，利用前缀树可以在每次询问对子区间内的每个元素在O(log n)的时间内得到答案，执行n此的时间花费为O(n logn)，而得到答案需要已经建立前缀树，而每次询问答案都需要重新建立一棵前缀树，每次建树最坏情况下的时间花费为O(n)。总的时间为O(n^2 logn)，对于这题来说已经足够了。

```c++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<vector<int>> son;
vector<int> a;
int idx;

void insert(int x){
    int p = 0;
    for(int i = 10; i >= 0; i--){
        int u = ((x >> i) & 1);
        if(!son[p][u]){
            son[p][u] = ++idx;
        }
        p = son[p][u];
    }
}

int query(int x){
    int p = 0;
    int ret = 0;
    for(int i = 10; i >= 0; i--){
        int u ((x >> i) & 1);
        if(son[p][!u]){
            ret = ret * 2 + 1;
            p = son[p][!u];
        }
        else{
            ret = ret * 2;
            p = son[p][u];
        }
    }
    return ret;
}

void solve(){
    int n, m;
    cin >> n >> m;
    a.resize(n + 1);
    for(int i = 1; i <= n; i++){
        cin >> a[i];
    }

    son.resize((n + 10) * 10, vector<int>(2));
    while(m--){
        idx = 0;
        int l, r;
        cin >> l >> r;
        for(int i = l; i <= r; i++){
            insert(a[i]);
        }

        int maxv = 0;
        for(int i = l; i <= r; i++){
            maxv = max(maxv, query(a[i]));
        }

        cout << maxv << endl;

        for(int i = 0; i < idx; i++){
            son[i][0] = son[i][1] = 0;
        }
    }

}

int main(){
    solve();
    return 0;
}
```

## [求子区间异或和最大值C. Vampiric Powers, anyone?](https://codeforces.com/problemset/problem/1847/C)

分析性质可以知道，只要求出子区间所能得到的最大异或和即可。而求一段连续子区间的异或和，可以利用前缀和：

s[n] = s[1] ^ s[2] ^ s[3] ^ ... ^ s[i] ^ s[i + 1] ^ ... ^ s[n]

s[i] = s[1] ^ s[2] ^ s[3] ^ ... ^ s[i] = s[n] ^ s[i - 1]

把这样处理得到的前缀和序列作为对象，求子区间异或和最大值的问题就被转换成了求区间异或对最大值问题。利用前缀树处理，就能在O(log n)的时间内得到子区间异或和的最大值。

```c++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<int> a;
vector<vector<int>> son;
int idx;

void insert(int x){
	int p = 0;
	for(int i = 8; i >= 0; i--){
		int u = ((x >> i) & 1);
		if(!son[p][u]){
			son[p][u] = ++idx;
		}
		p = son[p][u];
	}
}

int query(int x){
	int p = 0;
	int ret = 0;
	for(int i = 8; i >= 0; i--){
		int u = ((x >> i) & 1);
		if(son[p][!u]){
			ret = ret * 2 + 1;
			p = son[p][!u];
		}
		else{
			ret = ret * 2;
			p = son[p][u];
		}
	}
	return ret;
}

void solve(){
	idx = 0;
	int n;
	cin >> n;
	a.resize(n + 1);
	for(int i = 1; i <= n; i++){
		cin >> a[i];
		a[i] = a[i] ^ a[i - 1];
	}

	son.resize((n + 10) * 8, vector<int>(2));

	int maxv = 0;
	for(int i = 0; i <= n; i++){
		insert(a[i]);
		maxv = max(maxv, query(a[i]));
	}
	
	cout << maxv << endl;

	for(int i = 0; i <= idx; i++){
		son[i][0] = son[i][1] = 0;
	}
}

int main(){
	ios::sync_with_stdio(false);
	cin.tie(0);
	int t = 1;
	cin >> t;
	while(t--){
		solve();
	}
	return 0;
}
```



# 二叉搜索树

## [洛谷P3369 【模板】普通平衡树](https://www.luogu.com.cn/problem/P3369)

```c++
#include<bits/stdc++.h>
using namespace std;

vector<int> v;

void solve(){
    int opt, x;
    cin >> opt >> x;
    if(opt == 1){
        v.insert(lower_bound(v.begin(), v.end(), x), x);
    }
    if(opt == 2){
        v.erase(lower_bound(v.begin(), v.end(), x));
    }
    if (opt == 3){
        cout << lower_bound(v.begin(), v.end(), x) - v.begin() + 1 << endl;
    }
    if(opt == 4){
        cout << v[x - 1] << endl;
    }
    if(opt == 5){
        cout << v[lower_bound(v.begin(), v.end(), x) - v.begin() - 1] << endl;
    }
    if(opt == 6){
        cout << v[upper_bound(v.begin(), v.end(), x) - v.begin()] << endl;
    }
}

int main(){
    int n;
    cin >> n;
    while(n--){
        solve();
    }
    return 0;
}
```



# 扫描线

[知乎专栏](https://zhuanlan.zhihu.com/p/103616664)

[codeforces题目 rank1800](https://codeforces.com/problemset/problem/1859/D)

简单来说，就是整理出一系列待处理的状态，排序之后从头到尾，按照类型的不同进行不同的操作。



# 线段树板子

https://codeforces.com/contest/1899

https://codeforces.com/blog/entry/122407

```c++
#include <bits/stdc++.h>
 
using namespace std;
 
#define sz(x) (int)x.size()
#define all(x) x.begin(), x.end()
 
struct SegmentTree {
    int n;
    vector<vector<int>> tree;
 
    void build(vector<int> &a, int x, int l, int r) {
        if (l + 1 == r) {
            tree[x] = {a[l]};
            return;
        }
 
        int m = (l + r) / 2;
        build(a, 2 * x + 1, l, m);
        build(a, 2 * x + 2, m, r);
        merge(all(tree[2 * x + 1]), all(tree[2 * x + 2]), back_inserter(tree[x]));
    }
 
    SegmentTree(vector<int>& a) : n(a.size()) {
        int SIZE = 1 << (__lg(n) + bool(__builtin_popcount(n) - 1));
        tree.resize(2 * SIZE - 1);
        build(a, 0, 0, n);
    }
 
    int count(int lq, int rq, int mn, int mx, int x, int l, int r) {
        if (rq <= l || r <= lq) return 0;
        if (lq <= l && r <= rq) return lower_bound(all(tree[x]), mx) - lower_bound(all(tree[x]), mn);
 
        int m = (l + r) / 2;
        int a = count(lq, rq, mn, mx, 2 * x + 1, l, m);
        int b = count(lq, rq, mn, mx, 2 * x + 2, m, r);
        return a + b;
    }
 
    int count(int lq, int rq, int mn, int mx) {
        return count(lq, rq, mn, mx, 0, 0, n);
    }
};
 
vector<vector<int>> g;
 
vector<int> tin, tout;
int timer;
void dfs(int v, int p) {
    tin[v] = timer++;
    for (auto u : g[v]) {
        if (u != p) {
            dfs(u, v);
        }
    }
    tout[v] = timer;
}
 
void solve() {
    int n, q;
    cin >> n >> q;
    
    g.assign(n, vector<int>());
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        g[u].push_back(v);
        g[v].push_back(u);
    }
 
    timer = 0;
    tin.resize(n);
    tout.resize(n);
    dfs(0, -1);
    
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> p[i];
 
    vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = tin[p[i] - 1];
    SegmentTree ST(a);
 
    for (int i = 0; i < q; i++) {
        int l, r, x;
        cin >> l >> r >> x;
        l--; x--;
        if (ST.count(l, r, tin[x], tout[x])) {
            cout << "YES\n";
        } else {
            cout << "NO\n";
        }
    }
}
 
int main() {
    int tests;
    cin >> tests;
    while (tests--) {
        solve();
        if(tests > 0) cout << "\n";
    }
    return 0;
}
```



## 求连续子区间元素和的最大值

```c++
struct SegmentTree{
    struct Node{
        ll sum = 0, lsum = 0, rsum = 0, msum = 0;
    };

    vector<Node> tree;

    void push_up(int p){
        int lch = p * 2 + 1;
        int rch = p * 2 + 2;
        tree[p].sum = tree[lch].sum + tree[rch].sum;
        tree[p].lsum = max(tree[lch].lsum, tree[lch].sum + tree[rch].lsum);
        tree[p].rsum = max(tree[rch].rsum, tree[rch].sum + tree[lch].rsum);
        tree[p].msum = max({tree[lch].msum, tree[rch].msum, tree[lch].rsum + tree[rch].lsum});
    }

    void build(vector<ll> &a, int p, int l, int r){
        if(l == r){
            tree[p].sum = tree[p].lsum = tree[p].rsum = tree[p].msum = a[l];
            return;
        }
        int mid = (l + r) / 2;
        int lch = p * 2 + 1;
        int rch = p * 2 + 2;
        build(a, lch, l, mid);
        build(a, rch, mid + 1, r);
        push_up(p);
    }

    SegmentTree(vector<ll> &a){
        tree = vector<Node>(a.size() * 4, {0, 0, 0, 0});
        build(a, 0, 0, a.size() - 1);
    }

    void modify(int p, int i, int k, int l, int r){
        if(l == r){
            tree[p].sum = tree[p].lsum = tree[p].rsum = tree[p].msum = k;
            return;
        }
        int mid = (l + r) / 2;
        int lch = p * 2 + 1;
        int rch = p * 2 + 2;
        if(i <= mid){
            modify(lch, i, k, l, mid);
        }
        else{
            modify(rch, i, k, mid + 1, r);
        }
        push_up(p);
    }

    void modify(int i, int k){
        modify(0, i, k, 0, tree.size() / 4 - 1);
    }

    Node query(int p, int l, int r, int pl, int pr){
        if(l <= pl && pr <= r){
            return tree[p];
        }
        int mid = (pl + pr) / 2;
        int lch = p * 2 + 1;
        int rch = p * 2 + 2;
        Node ret;
        Node lnode;
        bool lfit = false;
        if(l <= mid){
            lnode = query(lch, l, r, pl, mid);
            ret = lnode;
            lfit = true;
        }
        if(r > mid){
            Node rnode = query(rch, l, r, mid + 1, pr);
            if(!lfit){
                ret = rnode;
            }
            else{
                ret.msum = max({lnode.msum, rnode.msum, lnode.rsum + rnode.lsum});
                ret.lsum = max(lnode.lsum, lnode.sum + rnode.lsum);
                ret.rsum = max(rnode.rsum, rnode.sum + lnode.rsum);
                ret.sum = lnode.sum + rnode.sum;
            }
        }
        return ret;
    }

    ll query(int l, int r){
        return query(0, l, r, 0, tree.size() / 4 - 1).msum;
    }
};
```



https://codeforces.com/contest/1925/problem/E

## LazySegmentTree

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

template<class Info, class Tag>
struct LazySegmentTree{
    int n;
    vector<Info> info;
    vector<Tag> tag;
    LazySegmentTree(): n(0){}
    LazySegmentTree(int n_, Info v_ = Info()){
        init(vector<Info>(n_, v_));
    }
    LazySegmentTree(vector<Info> init_){
        init(init_);
    }
    void init(vector<Info> init_){
        n = init_.size();
        info.assign(4 * n, Info());
        tag.assign(4 * n, Tag());
        function<void(int, int, int)> build = [&](int p, int l, int r){
            if(r - l == 1){
                info[p] = init_[l];
                return;
            }
            int m = (l + r) / 2;
            build(2 * p, l, m);
            build(2 * p + 1, m, r);
            pull(p);
        };
        build(1, 0, n);
    }
    void pull(int p){
        info[p] = info[2 * p] + info[2 * p + 1];
    }
    void apply(int p, const Tag &v){
        info[p].apply(v);
        tag[p].apply(v);
    }
    void push(int p){
        apply(2 * p, tag[p]);
        apply(2 * p + 1, tag[p]);
        tag[p] = Tag();
    }
    void modify(int p, int l, int r, int x, const Info &v){
        if(r - l == 1){
            info[p] = v;
            return;
        }
        int m = (l + r) / 2;
        push(p);
        if(x < m){
            modify(2 * p, l, m, x, v);
        }
        else{
            modify(2 * p + 1, m, r, x, v);
        }
        pull(p);
    }
    void modify(int x, const Info &v){
        modify(1, 0, n, x, v);
    }
    Info rangeQuery(int p, int l, int r, int x, int y){
        if(l >= y || r <= x){
            return Info();
        }
        if(l >= x && r <= y){
            return info[p];
        }
        int m = (l + r) / 2;
        push(p);
        return rangeQuery(2 * p, l, m, x, y) + rangeQuery(2 * p + 1, m, r, x, y);
    }
    Info rangeQuery(int l, int r){
        return rangeQuery(1, 0, n, l, r);
    }
    void rangeApply(int p, int l, int r, int x, int y, const Tag &v){
        if(l >= y || r <= x){
            return;
        }
        if(l >= x && r <= y){
            apply(p, v);
            return;
        }
        int m = (l + r) / 2;
        push(p);
        rangeApply(2 * p, l, m, x, y, v);
        rangeApply(2 * p + 1, m, r, x, y, v);
        pull(p);
    }
    void rangeApply(int l, int r, const Tag &v){
        return rangeApply(1, 0, n, l, r, v);
    }
};

struct Tag{
    ll k = 0;
    ll b = 0;

    void apply(const Tag &t){
        if(t.k < 0){
            k = t.k;
            b = t.b;
        }
    }
};

struct Info{
    ll cnt = 0;
    ll sid = 0;
    ll sum = 0;

    void apply(const Tag &t){
        if(t.k < 0){
            sum = sid * t.k + cnt * t.b;
        }
    }
};

Info operator + (const Info &a, const Info &b){
    return {a.cnt + b.cnt, a.sid + b.sid, a.sum + b.sum};
}

void solve(){
    int n, m, q;
    cin >> n >> m >> q;
    vector<int> x(m);
    for(int i = 0; i < m; i++){
        cin >> x[i];
        x[i]--;
    }
    map<int, int> mp;
    for(int i = 0; i < m; i++){
        cin >> mp[x[i]];
    }

    LazySegmentTree<Info, Tag> seg(n);
    for(int i = 0; i < n; i++){
        seg.modify(i, {1, i, 0});
    }
    
    auto work = [&](auto it){
        auto nxt = next(it);
        seg.rangeApply(it->first + 1, nxt->first + 1, {-it->second, (ll)it->second * nxt->first});
    };
    for(auto it = mp.begin(); it->first < n - 1; it++){
        work(it);
    }

    while(q--){
        int tp, x, y;
        cin >> tp >> x >> y;

        if(tp == 1){
            x--;
            mp[x] = y;
            auto it = mp.find(x);
            work(it);
            work(prev(it));
        }
        else{
            x--;
            ll ans = seg.rangeQuery(x, y).sum;
            cout << ans << "\n";
        }
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    solve();
    
    return 0;
}
```



# ordered_set

https://www.geeksforgeeks.org/ordered-set-gnu-c-pbds/



# 位运算

https://ac.nowcoder.com/acm/contest/67741/H

```c++
#include<bits/stdc++.h>

using namespace std;

typedef long long ll;
#define all(x) x.begin(), x.end()
#define rep(i, n) for(ll i = 0; i < (n); i++)
#define rep1(i, n) for(ll i = 1; i <= (n); i++)

void solve(){
    ll n, m;
    cin >> n >> m;
    vector<ll> v(n), w(n);
    rep(i, n){
        cin >> v[i] >> w[i];
    }

    auto work = [&](ll x){
        ll ret = 0;
        rep(i, n){
            if((x & w[i]) == w[i]){
                ret += v[i];
            }
        }
        return ret;
    };

    ll ans = work(m);
    for(ll x = m; x > 0; x-= (x & -x)){
        ll ret = work(x - 1);
        ans = max(ans, ret);
    }

    cout << ans << "\n";
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t = 1;
    cin >> t;
    while(t--){
        solve();
    }

    return 0;
}
```



# 双哈希

https://atcoder.jp/contests/abc339/tasks/abc339_f

```c++
#include<bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef unsigned long long ull;
#define all(x) x.begin(), x.end()
#define rep(i, n) for(ll i = 0; i < (n); i++)
#define rep1(i, n) for(ll i = 1; i <= (n); i++)

void solve(){
    ull n;
    cin >> n;
    vector<array<ull,2>> a(n);
    const ull mod = 1e9 + 7;
    map<array<ull,2>, ull> mp;
    rep(i, n){
        string s;
        cin >> s;
        ull e1 = 0;
        ull e2 = 0;
        rep(j, s.size()){
            e1 = e1 * 10 + (s[j] - '0');
            e2 = e2 * 10 + (s[j] - '0');
            e2 = (e2 % mod + mod) % mod;
        }
        a[i] = {e1, e2};
        mp[{e1, e2}]++;
    }

    ull ans = 0;
    rep(i, n){
        rep(j, n){
            ull e1 = a[i][0] * a[j][0];
            ull e2 = a[i][1] * a[j][1];
            e2 = (e2 % mod + mod) % mod;
            ans += mp[{e1, e2}];
        }
    }

    cout << ans << "\n";
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }

    return 0;
}
```


# 扩展域并查集

https://codeforces.com/gym/104901/problem/G

```c++
#include<bits/stdc++.h>

using namespace std;

typedef long long ll;
#define rep(i, n) for(ll i = 0; i < (n); i++)
#define rep1(i, n) for(ll i = 1; i <= (n); i++)
#define all(x) x.begin(), x.end()

struct Union{
    ll n;
    vector<ll> info;
    vector<bool> vis;

    Union(ll _n){
        n = _n;
        info.assign(n * 2, -1);
        vis.assign(n * 2, false);
        rep(i, 2 * n){
            info[i] = i;
        }
    }
    ll find_path(ll x){
        if(x != info[x]){
            info[x] = find_path(info[x]);
        }
        return info[x];
    }
    void join(ll x, ll y){
        vis[x] = true;
        vis[y] = true;
        x = find_path(x);
        y = find_path(y);
        info[x] = y;
        find_path(x);

    }
    ll count(){
        ll ret = 0;
        rep(i, n * 2){
            if(i < n && info[i] == info[i + n]){
                return -1;
            }
            if(vis[i] && find_path(i) == i){
                ret++;
            }
        }
        return ret / 2;
    }
    void visit(ll x){
        vis[x] = vis[x + n] = true;
    }
};

const ll mod = 1e9 + 7;

ll qpow(ll x, ll y){
    if(y == 0){
        return 1;
    }
    if(y % 2 == 0){
        return qpow(x * x % mod, y / 2) % mod;
    }
    else{
        return x * qpow(x * x % mod, y / 2) % mod;
    }
}

void solve(){
    ll r, c;
    cin >> r >> c;
    vector grid(r, vector<ll>(c, 0));
    rep(i, r){
        string s;
        cin >> s;
        rep(j, c){
            grid[i][j] = s[j] - '0';
        }
    }

    vector ocr(c, vector<ll>());
    Union un(r);
    ll blank = 0;
    rep(i, r){
        if(accumulate(all(grid[i]), 0) == 0){
            blank++;
            continue;
        }
        rep(j, c){
            if(grid[i][j]){
                if(ocr[j].size() > 0){
                    for(auto idx: ocr[j]){
                        un.join(i, idx + r);
                        un.join(i + r, idx);
                    }
                }
                if(ocr[c - 1 - j].size() > 0 ){
                    for(auto idx: ocr[c - 1 - j]){
                        un.join(i, idx);
                        un.join(i + r, idx + r);
                    }
                }
            }
        }
        rep(j, c){
            if(grid[i][j]){
                ocr[j].push_back(i);
                if(ocr[j].size() > 2){
                    cout << "0\n";
                    return;
                }
            }
        }
        un.visit(i);
    }

    ll n = un.count();

    if(n == -1){
        cout << "0\n";
    }
    else{
        n += blank;
        ll ans = qpow(2LL, n);
        cout << ans << "\n";
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t = 1;
    cin >> t;
    while(t--){
        solve();
    }

    return 0;
}
```



# 组合数

## 将m个相同的小球放入n个不同的盒子

https://codeforces.com/contest/1931/problem/G

$C_n^m$ 	-->	$$\binom{n}{m}$$

$$\binom{n + m - 1}{n - 1}$$  或  $$\binom{n + m - 1}{m}$$



## 第二类斯特林数 --> 将m个不同的小球放入n个相同的盒子



# __int128

## 输出

```c++
void write(__int128 x){
    if(x < 0){
        putchar('-');
        x = -x;
    }
    if(x > 9){
        write(x / 10);
    }
    putchar(x % 10 + '0');
}
```



# dijkstra

https://atcoder.jp/contests/abc342/tasks/abc342_e

```c++
#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
#define all(x) x.begin(), x.end()
#define rep(i, n) for(ll i = 0; i < (n); i++)
#define rep1(i, n) for(ll i = 1; i <= (n); i++)

void solve(){
    ll n, m;
    cin >> n >> m;
    struct Info{
        ll l, d, k, c, a, b;
    };
    vector<Info> a(m);
    rep(i, m){
        cin >> a[i].l >> a[i].d >> a[i].k >> a[i].c >> a[i].a >> a[i].b;
    }

    vector graph(n + 1, vector<array<ll,2>>());
    rep(i, m){
        graph[a[i].b].push_back({a[i].a, i});
    }
    const ll INF = 2e18;
    vector<ll> arr(n + 1, INF);
    priority_queue<array<ll,2>> pque;
    auto get_time = [&](int u, int v, int i){
        ll arrv = arr[u] - a[i].c;
        arrv = min((arrv - a[i].l) / a[i].d * a[i].d + a[i].l, a[i].l + (a[i].k - 1) * a[i].d);
        if(a[i].l <= arrv){
            return arrv;
        }
        else{
            return INF;
        }
    };
    vector<bool> vis(n + 1, false);
    auto add = [&](int u){
        vis[u] = true;
        for(auto [v, i]: graph[u]){
            if(!vis[v]){
                ll time = get_time(u, v, i);
                if(time != INF){
                    pque.push({get_time(u, v, i), v});
                }
            }
        }
    };

    add(n);
    while(!pque.empty()){
        auto [time, u] = pque.top();
        pque.pop();
        if(!vis[u]){
            arr[u] = time;
            vis[u] = true;
            add(u);
        }
    }

    rep1(i, n - 1){
        if(vis[i]){
            cout << arr[i] << "\n";
        }
        else{
            cout << "Unreachable\n";
        }
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    solve();

    return 0;
}
```

