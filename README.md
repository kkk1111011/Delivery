# AI1603-03-项目报告

## 第 XX 组
2024 年 7 月 17 日

### 小组成员
学号 523030910237 523030910196 523030910202 523030910205
姓名 方捷，孙羽恒，黄逸隽，张嘉欢

## 目录
1. [简介](#简介)
2. [策略实现方案及其可行性分析](#策略实现方案及其可行性分析)
    - [准备工作](#准备工作)
    - [价值分析和接送单策略](#价值分析和接送单策略)
        - [价值分析](#价值分析)
        - [简单的接单策略](#简单的接单策略)
        - [进阶的接单策略](#进阶的接单策略)
    - [动作空间的完成](#动作空间的完成)
3. [结果分析](#结果分析)
4. [结论](#结论)
5. [项目分工](#项目分工)

## 简介

本次大作业的目标是在一张给定的地图上，通过算法操控智能体在时间范围内尽可能多的将订单从餐厅送达至客户。

主要的规则如下：
在 NxN 的网格地图中（N = 16），地图中有 m 个餐厅（m = 10），k 个客户 (k = 20)，a 个骑手（a = 2）。地图中有道路。餐厅和客户的位置在游戏开始时在道路中随机产生，且每局游戏中位置不变。骑手初始位于道路的随机位置中，每步可以移动一个网格。地图如下图所示：

图 1: 初始地图

每个餐厅在时刻 t（t mod 10 = 0）随机产生的订单，每个餐厅的订单容量没有上限。若订单在到达规定送达时刻后仍未被骑手接受，该订单被自动取消。每隔一定时刻（10），系统会随机产生 20 个订单。每个时刻，系统会从还未被抢的订单中生成至多 10 个订单，骑手在每个时刻可以根据系统产生的订单进行抢单，在到达餐厅后根据已抢到的订单取单。每个骑手已抢单的数量上限为 20（不包含已取单的数量），已取单数量的上限为骑手的订单容量 cap（cap = 5）。骑手在订单规定的结束时间前将订单送至客户后可以获得奖励，奖励的值为餐厅到客户的距离（曼哈顿距离）。若骑手抢订单后未能在规定时间内取到订单或未能将订单在规定的时间内运送到客户（包括途中将订单丢弃），则获得惩罚，惩罚值为距离的一半。若订单未取到或未能在规定时刻送达客户，系统会将该订单从骑手订单中丢弃，骑手可重新获得一个订单容量。

订单包含的信息有订单产生的餐厅、下订单的客户和订单的结束时间（订单需在此时间之前被送到），订单的结束时间为当前时间加上餐厅到客户的距离值与 NxN 之间的一个随机值（random(distance(餐厅，客户), NxN)）。当订单被接受后，会产生取单时间，取单时间为当前时间加上当前骑手距餐厅的距离 D 与两倍 D 之间的一个随机值（random(distance(餐厅，骑手), 2 x distance(餐厅，骑手))）。

骑手的动作分为四个部分，分别为移动信息，取订单信息，放订单信息与抢订单信息，对应的动作空间为 [Discrete(5), Discrete(20), Discrete(5), Discrete(10)]。

本次大作业的策略就是根据观测值和一定的规划来完成骑手的动作空间，其目标是得到更多的分数（分数 = 奖励—惩罚）。

## 策略实现方案及其可行性分析

### 准备工作

在实现策略之前，我们首先要获取地图中的信息，这些信息储存在 observation 中。observation 为一个字典，其中的键为”obs” 和”controlled_player_index”。其中”obs”对应的值为一个字典，其中的键包括”agents”，”restaurants”，”customers”，”distributed_orders”和”roads”。”agents” 包括所有骑手的信息，如位置、订单等；”restaurants” 包含了所有餐厅的信息，如位置、订单等；”customers” 包含了所有客户的信息，如位置；”distributed_orders” 表示当前系统分配订单的结果；”roads” 为地图中的道路。

因此我们在预处理中定义了如下变量：
- `roads, cus_loc`: 存储道路、客户和餐馆的位置；
- `grid_map, idx_map, dis, dis_pre, dis_onmap`: 用于存储网格地图、索引地图和预计算的距离矩阵；
- `dx, dy`: 用于表示移动方向的偏移量；
- `players_score, players_state, players_ords_dict, players_pre_catched_ords, players_confirmed_ords, players_walk_path`: 用于存储每个玩家的状态、分数、订单信息等；

然后我们定义了两个函数来对地图信息进行预处理：

```python
def load_roads(obs):#读入地图信息
def calc_dis() : # 分别计算图中餐厅到客户之间的距离
```

经过这些基础处理之后，我们开始设计接单送单的策略

价值分析和接送单策略
价值分析
首先我们需要对当前系统产生的单子进行价值分析，为此我们定义了

$value = \frac{reward}{time}$
从而通过 value 这一指标来判断单子的价值高低，若当前系统无可接单子（时间不可行）则直接返回错误值-1

### 简单的接单策略
最简单的接单策略就是一次只接一单，我们在当前时刻产生的所有单子中选出价值最高且可送的订单，然后规划路径，送完该单后才接下一单。这是一种贪心算法的思路，但可能并没有考虑到全局最优，因此后续我们对其进行了优化，其详细代码如下：

```python
def reward_ord(res_pos, cus_pos):#算奖励
    return abs(cus_pos[0] − res_pos[0]) + abs(cus_pos[1] − res_pos[1])

def time_spent(res_pos, cus_pos, pos):#算耗时
    dis_to_res = dis_onmap[pos[0]][pos[1]][res_pos [0]][ res_pos[1]]
    dis_to_cus = dis_onmap[res_pos[0]][res_pos[1]][cus_pos[0]][cus_pos[1]]
    dis_tot = dis_to_res + dis_to_cus
    return dis_tot

def calc_value(s_ord, pos, cur):#计算单个订单的价值
    ddl_time = s_ord[’end_time’] − cur
    res_id = s_ord[’restaurant_id’]
    cus_id = s_ord[’customer_id’]
    res_pos = res_loc[res_id]
    cus_pos = cus_loc[cus_id]
    time_need = time_spent(res_pos, cus_pos, pos)
    ok_sent = (time_need + 2) <= ddl_time # +2 is Edge buffering // Rider can have a rest ......
    reward = reward_ord(res_pos, cus_pos)
    return [ok_sent ∗ reward / time_need, reward] # [value, reward]

def choose_an_ord(new_ords, pos, cur): # pos = player’s position #选择一个订单
    if len(new_ords) == 0: return −1
    # value(ord) = can_be_sent(ord) ∗ reward(ord) / time_need(ord)
    val_lis = []
    for i in range(len(new_ords)):
        val_lis .append([i, calc_value(new_ords[i], pos, cur) ])
    val_lis . sort(key=lambda x:x[1][0], reverse=True)
    return [ val_lis [0][0], val_lis [0][1]] # [idx, [value , reward]]
```
### 进阶的接单策略
在初级接单策略中我们一次只接一单，这忽略了在系统当前给出的订单中可能有组合的价值高于只接一单。为了能够同时考虑更多的单子，我们要对订单进行筛选、排序和路径规划。我们采用枚举的方法来实现这一目的。

首先我们定义了两个辅助函数：

```python
def generate_permutations(n, num_ones):#生成01全排列
def is_valid_perm(perm):#判断排列（餐厅与客户在路径上的顺序）是否合法，如果访问客户在商家之前则不合法
```

接下来进行订单的选取，我们首先生成当前订单的全组合（参数 `maxpick` 用于限制选出单子的数目），找到其中价值最高的组合，这个组合中，预备要选的订单对应下标记作 `1`，不选订单记作 `0`。我们对所有订单生成 01 排列，然后筛选出其中合法的排列（通过 `is_valid_perm` 判断），找到合法排列中的最佳方案。代码如下：

```python
def pick_ords(new_ords, pos, cur, maxpick=2): # choose maximum of maxpick orders
    if len(new_ords) == 0: return −1
    ord_idx = list(range(len(new_ords)))
    perm = generate_permutations(len(new_ords), maxpick)
    val_perm = []
    for each_perm in perm:
        if is_valid_perm(each_perm):
            val_perm.append(each_perm)
    best_value = -1
    best_pick = -1
    for each_perm in val_perm:
        time_need = 0
        reward = 0
        value = 0
        for i in range(len(each_perm)):
            if each_perm[i] == 1:
                s_ord = new_ords[ord_idx[i]]
                res_id = s_ord['restaurant_id']
                cus_id = s_ord['customer_id']
                res_pos = res_loc[res_id]
                cus_pos = cus_loc[cus_id]
                time_need += time_spent(res_pos, cus_pos, pos)
                reward += reward_ord(res_pos, cus_pos)
                pos = cus_pos
        value = reward / time_need
        if value > best_value:
            best_value = value
            best_pick = each_perm
    return best_pick
```
## 动作空间的完成
为了完成动作空间的实现，我们将接单策略嵌入到骑手的行动中。在每个时刻，骑手根据当前状态（位置、订单、时间）选择最佳动作（移动、取单、送单、抢单）。通过对接单策略的优化，我们可以让骑手在有限的时间内完成尽可能多的订单，从而最大化得分。最终的策略实现包括：

1. 移动策略：骑手在当前时刻根据最优路径规划移动，避免拥堵和绕路。
2. 取单策略：根据接单策略，骑手选择价值最高的订单进行取单。
3. 送单策略：在送单过程中，骑手根据最短路径将订单送至客户。
4. 抢单策略：在系统产生新订单时，骑手根据订单的价值和时间限制进行抢单。
通过上述策略的实现，我们可以确保骑手在整个时间范围内高效地完成订单，获得最大化的奖励。

## 结论
本算法在框架上基本实现任务所需目标，也达到了比较好的效果。但是对于中间的
一些处理可能仍有优化空间，以下是一些优化的想法：
1. 在抢单策略中，max_pick 我们最终定为 3，这是由于枚举算法的复杂度导致的，
枚举 n=5 的时候数量级已经达到 106̂，为此我们可以考虑对用户和商家进行聚类，模
拟现实中的分区规划，餐饮片区和住宅区，提高处理速度，从而避免了对离散的点进行
暴力枚举。
2. 我们采取的是送完当前单再继续接单的算法，事实上可以通过动态规划的方法
在送单的同时继续接单，可能可以取得全局的更优。
3. 在抢送单策略中没有考虑舍弃订单的问题，可能可以通过舍弃一些已接的单子
实现奖励最大化。
## 项目分工
- 张嘉欢：程序设计
- 方捷：报告撰写
- 黄逸隽、孙羽恒: 海报展示及课堂展示
