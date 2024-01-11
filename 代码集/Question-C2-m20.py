
# 第三方库
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
movies= pd.read_csv('mll.movies.csv')
rate=pd.read_csv('mll.ratings.csv')
# 合并数据
data = pd.merge(rate, movies, on='movieId')
data

rate_bar=rate["rating"].value_counts()
#绘制柱状图，直观判断可作为推荐电影的评分标准
plt.bar(rate_bar.index,rate_bar.values,width=0.3,color='blue')  
plt.xlabel('电影评分')#设置x轴标签  
plt.ylabel('统计次数')#设置y轴标签  
plt.title('电影评分柱状图')#设置图表标题  
plt.legend(['评分'])#添加图例
#在柱状图上方显示统计的次数
for i in range(len(rate_bar)):
    plt.text(rate_bar.index[i],rate_bar.values[i],str(rate_bar.values[i]),ha='center',va='bottom')
plt.xticks(rate_bar.index,rotation=0)#x轴显示  
plt.show()


#本案例进行电影推荐，考虑评分都小于4分的电影没有作为被推荐电影的潜质
support_rate=rate[rate['rating']>=4]
print(support_rate.head(15))

#删去时间戳timestamp和评分rating，方便进行频繁项集的统计
user_movie=support_rate[['userId','movieId']]
#统计列数据的频率分布
movie_counts=user_movie['movieId'].value_counts()
movie_counts_values=movie_counts.values.tolist()
print(f"{movie_counts.head(10)}\n")
print("共有{}份电影拥有作为被推荐电影的潜质\n".format(len(movie_counts_values)))


#认为优评过少不足以进入推荐电影的名单，优评次数大于75次的电影视作推荐电影
filtered_values=movie_counts[movie_counts>75].index
#根据筛选后的值构建新表
final_movie=user_movie[user_movie['movieId'].isin(filtered_values)]
#输出新表
print(final_movie)


#将DataFrame转换为字典形式，统计每个用户更推荐哪些电影（优评数大于75次）
userId_dict=final_movie.groupby('userId')['movieId'].agg(list).to_dict()
#将字典的值转换为列表，每个元素也是一个列表
transactions=[m for n,m in userId_dict.items()]
for item in transactions:
    print(item)


#Apriori算法的主函数
def apriori(data,min_support):
   #创建候选项集C1
    C1=create_C1(data)
   #将数据集的每一项转换为集合
    D=list(map(set,data))
   #扫描数据集D，生成频繁项集L1
    L1,support_data=scan_D(D,C1,min_support)
   #初始化频繁项集列表L，包含L1
    L=[L1]
    k=2
   #初始化频繁项集列表
    frequent_itemsets=[]
    while len(L[k-2])>0:
       #使用Apriori-gen函数生成候选项集Ck
        Ck=apriori_gen(L[k-2],k)
       #扫描数据集D，生成频繁项集Lk
        Lk,supK=scan_D(D,Ck,min_support)
       #更新支持度字典
        support_data.update(supK)
       #添加频繁项集Lk到L
        L.append(Lk)
        k+=1
       #将满足条件的频繁项集添加到frequent_itemsets
        for itemset in Lk:
            if support_data[itemset]>=min_support:
                frequent_itemsets.append(itemset)
    return L,support_data,frequent_itemsets

#创建候选项集C1
def create_C1(data):
    C1=[]
   #遍历数据集中的每一项transaction
    for transaction in data:
       #遍历transaction中的每一项item
        for item in transaction:
           #如果项集[item]不在C1中，则添加到C1
            if not [item] in C1:
                C1.append([item])
   #对C1进行排序
    C1.sort()
   #使用frozenset，使C1中的每个项集都是不可变的
    return list(map(frozenset,C1))

#扫描数据集D，生成频繁项集Lk
def scan_D(D,Ck,min_support):
    ssCnt={}
   #遍历数据集中的每一项tid
    for tid in D:
       #遍历候选项集Ck
        for can in Ck:
           #检查候选项集can是否是tid的子集
            if can.issubset(tid):
               #如果是，则增加候选项集can的计数
                if not can in ssCnt:ssCnt[can]=1
                else:ssCnt[can]+=1
    num_items=float(len(D))
    retList=[]
    support_data={}
   #遍历候选项集的支持度字典ssCnt
    for key in ssCnt:
       #计算候选项集的支持度
        support=ssCnt[key]/num_items
       #如果支持度大于最小支持度，则将候选项集添加到频繁项集列表
        if support>=min_support:
            retList.insert(0,key)
       #将候选项集的支持度添加到支持度字典
        support_data[key]=support
   #返回频繁项集列表和支持度字典
    return retList,support_data

#使用Apriori-gen函数生成候选项集Ck
def apriori_gen(Lk,k):
    retList=[]
    lenLk=len(Lk)
   #遍历频繁项集列表Lk中的每一项
    for i in range(lenLk):
       #遍历Lk中的剩余项
        for j in range(i+1,lenLk):
           #取两个频繁项集的前k-2个元素
            L1=list(Lk[i])[:k-2]; L2=list(Lk[j])[:k-2]
           #对元素进行排序
            L1.sort(); L2.sort()
           #如果前k-2个元素相同，则将两个集合合并
            if L1 ==L2:
                retList.append(Lk[i] | Lk[j])
   #返回候选项集列表
    return retList


#定义一个树节点类
class TreeNode:
    def __init__(self,name_value,num_occur,parent_node):
        self.name=name_value #节点名称
        self.count=num_occur #节点出现次数
        self.node_link=None #用于链接相似的元素项
        self.parent=parent_node #指向父节点
        self.children={} #存储子节点

   #增加节点的计数值
    def inc(self,num_occur):
        self.count+=num_occur

#创建FP树
def create_tree(data_set,min_sup=1):
    header_table={} #头指针表
   #第一次遍历数据集，创建头指针表
    for trans in data_set:
        for item in trans:
            header_table[item]=header_table.get(item,0) + data_set[trans]
   #移除不满足最小支持度的元素项
    header_table={k: v for k,v in header_table.items() if v>= min_sup}
    freq_item_set=set(header_table.keys()) #保存频繁项集
    if len(freq_item_set) == 0: #如果没有元素项满足要求，则退出
        return None,None
    for k in header_table:
        header_table[k]=[header_table[k],None] #初始化头指针表
    ret_tree=TreeNode('Null Set',1,None) #初始化FP树
   #第二次遍历数据集，创建FP树
    for tran_set,count in data_set.items():
        local_data={} #记录频繁1项集的全局频率，用于排序
        for item in tran_set: #根据全局频率对每个事务中的元素进行排序
            if item in freq_item_set:
                local_data[item]=header_table[item][0]
        if len(local_data)>0:
            ordered_items=[v[0] for v in sorted(local_data.items(),key=lambda p: p[1],reverse=True)]
            update_tree(ordered_items,ret_tree,header_table,count) #使用排序后的频繁项集对树进行填充
    return ret_tree,header_table

#更新FP树
def update_tree(items,in_tree,header_table,count):
    if items[0] in in_tree.children: #检查事务中的第一个元素项是否作为子节点存在
        in_tree.children[items[0]].inc(count) #如果存在，则增加该元素项的计数
    else: #如果不存在，则创建一个新的TreeNode并将其作为一个子节点添加到树中
        in_tree.children[items[0]]=TreeNode(items[0],count,in_tree)
        if header_table[items[0]][1] is None: #更新头指针表或前一个相似元素项节点的指针指向新节点
            header_table[items[0]][1]=in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1],in_tree.children[items[0]])
    if len(items)>1: #对剩下的元素项迭代调用update_tree函数
        update_tree(items[1::],in_tree.children[items[0]],header_table,count)

#更新头指针表
def update_header(node_to_test,target_node):
    while node_to_test.node_link is not None: #不断迭代直到当前节点的node_link指向None
        node_to_test=node_to_test.node_link
    node_to_test.node_link=target_node #添加到链表的末尾

#发现以给定元素项结尾的所有路径函数
def ascend_tree(leaf_node,prefix_path):
    if leaf_node.parent is not None: #递归上溯整棵树
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent,prefix_path)

#发现条件模式基
def find_prefix_path(base_pat,tree_node):
    cond_pats={} #条件模式基字典
    while tree_node is not None:
        prefix_path=[] #前缀路径
        ascend_tree(tree_node,prefix_path) #寻找当前非空节点的前缀路径
        if len(prefix_path)>1:
            cond_pats[frozenset(prefix_path[1:])]=tree_node.count #将条件模式基添加到字典中
        tree_node=tree_node.node_link #进入下一个相似节点
    return cond_pats

#递归查找频繁项集
def mine_tree(in_tree,header_table,min_sup,pre_fix,freq_item_list):
    big_l=[v[0] for v in sorted(header_table.items(),key=lambda p: p[1][0])] #对头指针表中的元素项按照频繁度排序，得到频繁项列表
    for base_pat in big_l: #从头指针表的底端开始
        new_freq_set=pre_fix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set) #将每个频繁项添加到频繁项集列表中
        cond_patt_bases=find_prefix_path(base_pat,header_table[base_pat][1]) #创建条件基
        my_cond_tree,my_head=create_tree(cond_patt_bases,min_sup) #创建条件FP树
        if my_head is not None: #挖掘条件FP树
            mine_tree(my_cond_tree,my_head,min_sup,new_freq_set,freq_item_list)

#FP-Growth算法主函数
def fp_growth(data_set,min_sup=1):
    init_set=create_init_set(data_set) #创建初始集
    my_fp_tree,my_header_tab=create_tree(init_set,min_sup) #创建FP树
    freq_items=[] #频繁项集列表
    mine_tree(my_fp_tree,my_header_tab,min_sup,set([]),freq_items) #创建条件FP树
   #按照频繁项集内元素的个数从小到大排序
    freq_items.sort(key=len)
   #过滤掉频繁一项集
    freq_items=[item for item in freq_items if len(item)>1]
    return freq_items #返回频繁项集列表

#创建初始集
def create_init_set(data_set):
    ret_dict={} #初始集字典
    for trans in data_set:
        ret_dict[frozenset(trans)]=1
    return ret_dict


minsup =100
start_time = time.perf_counter()
freq_itemsets_fp = fp_growth(transactions, minsup)
end_time = time.perf_counter()
for itemset in freq_itemsets_fp:
    print(f'frozenset({itemset})')
# 计算函数运行时间（结束时间 - 开始时间），并打印结果
print(f'这个函数一共运行了{end_time - start_time:.4f}秒')


#设置最小支持度
min_support=0.17
start_time = time.perf_counter()
#调用Apriori函数
L,support_data,frequent_itemsets_apriori=apriori(transactions,min_support)
end_time = time.perf_counter()
#打印频繁项集
for itemset in frequent_itemsets_apriori:
    print(itemset)
# 计算函数运行时间（结束时间 - 开始时间），并打印结果
print(f'这个函数一共运行了{end_time - start_time:.4f}秒')


def generate_rules(frequent_itemsets,support_data,min_confidence=0.7):
    big_rule_list=[]
   #遍历所有的频繁项集
    for freq_set in frequent_itemsets:
       #只考虑频繁项集大小大于1的情况
        if len(freq_set)>1:
            H1=[frozenset([item]) for item in freq_set]
            rules_from_conseq(freq_set,H1,support_data,big_rule_list,min_confidence)
            calc_confidence(freq_set,H1,support_data,big_rule_list,min_confidence)
   #按置信度从高到低排序
    big_rule_list.sort(key=lambda x:x[2],reverse=True)
    return big_rule_list

def calc_confidence(freq_set,H,support_data,brl,min_confidence=0.7):
    pruned_H=[]
    for conseq in H:
        conf=support_data[freq_set]/support_data[freq_set-conseq]
        if conf >= min_confidence:
            lift=conf/support_data[conseq] #计算提升度
            print(freq_set-conseq,'-->',conseq,'conf:',conf,'lift:',lift)
            brl.append((freq_set-conseq,conseq,conf,lift)) #将提升度添加到规则列表
            pruned_H.append(conseq)
    return pruned_H

def rules_from_conseq(freq_set,H,support_data,brl,min_confidence=0.7):
    m=len(H[0])
    if len(freq_set)>m+1:
        Hmp1=apriori_gen(H,m+1)
        Hmp1=calc_confidence(freq_set,Hmp1,support_data,brl,min_confidence)
        if len(Hmp1)>1:
            rules_from_conseq(freq_set,Hmp1,support_data,brl,min_confidence)


#设置最小置信度
min_confidence=0.85
apriori_rules=generate_rules(frequent_itemsets_apriori,support_data,min_confidence)

def rules_with_movie_names(rules,movie):
   #创建一个字典，将电影id映射到电影名称
    movie_dict=pd.Series(movie.title.values,index=movie.movieId).to_dict()
    for rule in rules:
       #将电影id转换为电影名称
        antecedent=[movie_dict[i] for i in rule[0]]
        consequent=[movie_dict[i] for i in rule[1]]
        confidence=rule[2]
        lift=rule[3] #提升度
       #按照指定的格式输出关联规则
        print("规则：如果一个人喜欢电影{}，那么他可能喜欢电影{}。这个规则的置信度为{:.2f}，提升度为{:.2f}\n".format(antecedent,consequent,confidence,lift))

#导入电影id与电影名称的对照表
rules_with_movie_names(apriori_rules,movies)

#设置隐因子数为k
k=40
#设置迭代次数为iterations
iterations=10
#设置学习率为learning_rate
learning_rate=0.01
#设置正则化率为regularization
regularization=0.01

def round_half(rating):
    return np.round(rating * 2) / 2

def truncate(rating):
    if rating > 5:
        return 5
    elif rating < 0.5:
        return 0.5
    else:
        return rating


def funk_svd(df,k, iterations, learning_rate, regularization):

    # 初始化存储结果的列表
    rmse_list = []
    mae_list = []
    time_list = []

    # 获取用户和电影的数量
    m = len(df['userId'].unique())
    n = len(df['movieId'].unique())

    # 初始化用户特征矩阵P和物品特征矩阵Q
    P = np.random.normal(scale=1./k, size=(m, k))
    Q = np.random.normal(scale=1./k, size=(n, k))

    # 将用户和电影的id转换为整数索引
    df['userId'] = pd.Categorical(df['userId']).codes
    df['movieId'] = pd.Categorical(df['movieId']).codes

    # 循环iterations次
    for i in range(iterations):
        start_time = time.time()  # 开始计时

        # 遍历每一行数据
        for index, row in df.iterrows():
            user, movie, rating = int(row['userId']), int(row['movieId']), row['rating']
            prediction = np.dot(P[user, :], Q[movie, :])  # 计算预测评分
            prediction = round_half(prediction)
            prediction = truncate(prediction)
            error = rating - prediction  # 计算误差

            # 更新P和Q
            P[user, :] += learning_rate * (error * Q[movie, :] - regularization * P[user, :])
            Q[movie, :] += learning_rate * (error * P[user, :] - regularization * Q[movie, :])

        # 计算整体的预测评分矩阵
        all_user_predicted_ratings = np.dot(P, Q.T)

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(df['rating'], all_user_predicted_ratings[df['userId'], df['movieId']]))
        rmse_list.append(rmse)

        # 计算MAE
        mae = mean_absolute_error(df['rating'], all_user_predicted_ratings[df['userId'], df['movieId']])
        mae_list.append(mae)

        end_time = time.time()  # 结束计时
        time_list.append(end_time - start_time)  # 计算运行时间

        # 输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    return rmse_list, mae_list, time_list

# 调用函数
rmse_list_funk_svd, mae_list_funk_svd, time_list = funk_svd(rate,k,iterations,learning_rate,regularization)
# 输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


def bias_svd(df, k, iterations, learning_rate, regularization):

    # 初始化存储结果的列表
    rmse_list = []
    mae_list = []
    time_list = []

    # 获取用户和电影的数量
    m = len(df['userId'].unique())
    n = len(df['movieId'].unique())

    # 初始化用户特征矩阵P和物品特征矩阵Q
    P = np.random.normal(scale=1./k, size=(m, k))
    Q = np.random.normal(scale=1./k, size=(n, k))

    # 初始化用户偏置向量bu和物品偏置向量bi
    bu = np.zeros(m)
    bi = np.zeros(n)

    # 计算全局平均评分
    global_mean = df['rating'].mean()

    # 将用户和电影的id转换为整数索引
    df['userId'] = pd.Categorical(df['userId']).codes
    df['movieId'] = pd.Categorical(df['movieId']).codes

    # 循环iterations次
    for i in range(iterations):
        start_time = time.time()  # 开始计时

        # 遍历每一行数据
        for index, row in df.iterrows():
            user, movie, rating = int(row['userId']), int(row['movieId']), row['rating']
            prediction = global_mean + bu[user] + bi[movie] + np.dot(P[user, :], Q[movie, :])  # 计算预测评分
            prediction = round_half(prediction)
            prediction = truncate(prediction)
            error = rating - prediction  # 计算误差

            # 更新P、Q、bu和bi
            P[user, :] += learning_rate * (error * Q[movie, :] - regularization * P[user, :])
            Q[movie, :] += learning_rate * (error * P[user, :] - regularization * Q[movie, :])
            bu[user] += learning_rate * (error - regularization * bu[user])
            bi[movie] += learning_rate * (error - regularization * bi[movie])

        # 计算整体的预测评分矩阵
        all_user_predicted_ratings = global_mean + bu[:, np.newaxis] + bi[np.newaxis, :] + np.dot(P, Q.T)

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(df['rating'], all_user_predicted_ratings[df['userId'], df['movieId']]))
        rmse_list.append(rmse)

        # 计算MAE
        mae = mean_absolute_error(df['rating'], all_user_predicted_ratings[df['userId'], df['movieId']])
        mae_list.append(mae)

        end_time = time.time()  # 结束计时
        time_list.append(end_time - start_time)  # 计算运行时间

        # 输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    return rmse_list, mae_list, time_list


# 调用函数
rmse_list_bias_svd, mae_list_bias_svd, time_list = bias_svd(rate,k, iterations, learning_rate, regularization)
# 输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


def svdpp(df, k, iterations, learning_rate, regularization):
    #初始化存储结果的列表
    rmse_list = []
    mae_list = []
    time_list = []

    # 获取用户和电影的数量
    m = len(df['userId'].unique())
    n = len(df['movieId'].unique())

    # 初始化用户特征矩阵P和物品特征矩阵Q
    P = np.random.normal(scale=1./k, size=(m, k))
    Q = np.random.normal(scale=1./k, size=(n, k))

    # 初始化用户偏置向量bu和物品偏置向量bi
    bu = np.zeros(m)
    bi = np.zeros(n)

    # 初始化隐式反馈矩阵Y
    Y = np.random.normal(scale=1./k, size=(n, k))

    # 计算全局平均评分
    global_mean = df['rating'].mean()

    # 创建一个字典，存储每个用户评分过的电影
    user_rated_movies = df.groupby('userId')['movieId'].apply(list).to_dict()

    # 创建一个映射，将movieId映射到一个连续的索引
    movieId_to_index = {movieId: index for index, movieId in enumerate(df['movieId'].unique())}

    # 循环iterations次
    for i in range(iterations):
        start_time = time.time()  # 开始计时

        # 遍历每一行数据
        for index, row in df.iterrows():
            user, movie, rating = int(row['userId']) - 1, movieId_to_index[row['movieId']], row['rating']  # 使用映射后的索引，并将userId减1
            rated_movies = [movieId_to_index[movieId] for movieId in user_rated_movies[user + 1]]  # 使用映射后的索引，这里需要将userId加回1
            sqrt_N_u = np.sqrt(len(rated_movies))

            # 计算隐式反馈的总和
            sum_Y = np.sum(Y[rated_movies], axis=0)

            # 计算预测评分
            prediction = global_mean + bu[user] + bi[movie] + np.dot(P[user, :] + sum_Y / sqrt_N_u, Q[movie, :])
            prediction = round_half(prediction)
            prediction = truncate(prediction)
            error = rating - prediction  # 计算误差

            # 更新P、Q、bu、bi和Y
            bu[user] += learning_rate * (error - regularization * bu[user])
            bi[movie] += learning_rate * (error - regularization * bi[movie])
            P[user, :] += learning_rate * (error * Q[movie, :] - regularization * P[user, :])
            Q[movie, :] += learning_rate * (error * (P[user, :] + sum_Y / sqrt_N_u) - regularization * Q[movie, :])
            Y[rated_movies] += learning_rate * (error * Q[movie, :] / sqrt_N_u - regularization * Y[rated_movies])

        # 计算整体的预测评分矩阵
        all_user_predicted_ratings = global_mean + bu[:, np.newaxis] + bi[np.newaxis, :] + np.dot(P, Q.T)
        # 将矩阵中的每个元素四舍五入到最近的0.5
        rounded_ratings = np.round(all_user_predicted_ratings * 2) / 2
        # 使用clip函数将所有小于0.5的值设置为0.5，所有大于5的值设置为5
        final_ratings = np.clip(rounded_ratings, 0.5, 5)
        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(df['rating'], all_user_predicted_ratings[df['userId'] - 1, [movieId_to_index[movieId] for movieId in df['movieId']]]))  # 使用映射后的索引，并将userId减1
        rmse_list.append(rmse)
        
        # 计算MAE
        mae = mean_absolute_error(df['rating'], all_user_predicted_ratings[df['userId'] - 1, [movieId_to_index[movieId] for movieId in df['movieId']]])
        mae_list.append(mae)

        end_time = time.time()  # 结束计时
        time_list.append(end_time - start_time)  # 计算运行时间

        # 输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    # 创建一个新的DataFrame，其中行索引为实际的userId，列名为在df中有对应评分的电影标题
    titles = df['title'].unique()
    if len(titles) < n:
        titles = np.append(titles, ['(UnknownMovie)'] * (n - len(titles)))
    final_ratings_df = pd.DataFrame(final_ratings, index=df['userId'].unique(), columns=titles)

    return rmse_list, mae_list, time_list, final_ratings_df


# 调用函数
rmse_list_svdpp, mae_list_svdpp, time_list,all_user_predicted_ratings = svdpp(data,k, iterations, learning_rate, regularization)

# 输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


x = range(1, 11)
y_funk_svd_rmse = rmse_list_funk_svd
y_bias_svd_rmse = rmse_list_bias_svd
y_svdpp_rmse = rmse_list_svdpp

y_funk_svd_mae = mae_list_funk_svd
y_bias_svd_mae = mae_list_bias_svd
y_svdpp_mae = mae_list_svdpp

# 绘制RMSE折线图
plt.figure(figsize=(10, 5))
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.plot(range(1, 11), rmse_list_funk_svd, label='Funk-SVD')
plt.plot(range(1, 11), rmse_list_bias_svd, label='BiasSVD')
plt.plot(range(1, 11), rmse_list_svdpp, label='SVD++')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
# 标记每一个点
plt.scatter(x, y_funk_svd_rmse, marker='*', color='blue')
plt.scatter(x, y_bias_svd_rmse, marker='D', color='orange')
plt.scatter(x, y_svdpp_rmse, marker='P', color='green')
for i in [0, 4]:
    plt.annotate(f"{y_funk_svd_rmse[i]:.3f}", (x[i], y_funk_svd_rmse[i]), textcoords="offset points", xytext=(0,2), ha='center', color='blue', fontsize=9)
    plt.annotate(f"{y_bias_svd_rmse[i]:.3f}", (x[i], y_bias_svd_rmse[i]), textcoords="offset points", xytext=(0,3), ha='center', color='black', fontsize=9)
    plt.annotate(f"{y_svdpp_rmse[i]:.3f}", (x[i], y_svdpp_rmse[i]), textcoords="offset points", xytext=(0,-10), ha='center', color='green', fontsize=9)
# 在背景中加入方格
plt.grid(linestyle='--', linewidth=1,alpha=0.3)
plt.legend()
plt.show()

# 绘制MAE折线图
plt.figure(figsize=(10, 5))
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.plot(range(1, 11), mae_list_funk_svd, label='Funk-SVD')
plt.plot(range(1, 11), mae_list_bias_svd, label='BiasSVD')
plt.plot(range(1, 11), mae_list_svdpp, label='SVD++')
plt.xlabel('Iteration')
plt.ylabel('MAE')
# 标记每一个点
plt.scatter(x, y_funk_svd_mae, marker='*', color='blue')
plt.scatter(x, y_bias_svd_mae, marker='D', color='orange')
plt.scatter(x, y_svdpp_mae, marker='P', color='green')
for i in [0, 4]:
    plt.annotate(f"{y_funk_svd_mae[i]:.3f}", (x[i], y_funk_svd_mae[i]), textcoords="offset points", xytext=(0,2), ha='center', color='blue', fontsize=8)
    plt.annotate(f"{y_bias_svd_mae[i]:.3f}", (x[i], y_bias_svd_mae[i]), textcoords="offset points", xytext=(0,2), ha='center', color='black', fontsize=8)
    plt.annotate(f"{y_svdpp_mae[i]:.3f}", (x[i], y_svdpp_mae[i]), textcoords="offset points", xytext=(0,-10), ha='center', color='green', fontsize=8)
# 在背景中加入方格
plt.grid(linestyle='--', linewidth=1,alpha=0.3)
plt.legend()
plt.show()



def recommend_movies(user_id, all_user_predicted_ratings, num_recommendations):
    # 获取用户的预测评分
    user_predicted_ratings = all_user_predicted_ratings.loc[user_id]
    # 获取评分最高的电影的名称
    recommended_movies = user_predicted_ratings.nlargest(num_recommendations).index.tolist()
    return recommended_movies


# 假设你想为用户推荐10部电影
userId=5
num=10
recommended_movies = recommend_movies(userId, all_user_predicted_ratings,num)
print(f"对用户ID为{userId}号的用户，我们推荐以下10个电影：{'| '.join(recommended_movies)}")


# 将补全的评分矩阵保存到文件
all_user_predicted_ratings.to_csv("mll.predicted_ratings.csv")






