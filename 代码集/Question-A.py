import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
movies=pd.read_csv('movies.csv')
rate=pd.read_csv('ratings.csv')
#观察数据集可知评分rating由0.5分-5分，0.5分为一档
print(rate.head(15))

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
print(f"{movie_counts_values}\n")
print("共有{}份电影拥有作为被推荐电影的潜质\n".format(len(movie_counts_values)))


#设置柱状图分段，观察各电影优评的频数分布
bins=[0,25,50,75,100,150,200,300]
#按分段离散化数据
segments=pd.cut(movie_counts_values,bins,right=False)
#统计各分段人数
counts=pd.value_counts(segments,sort=False)
#绘制柱状图
bar=plt.bar(counts.index.astype(str),counts)
plt.bar_label(bar,counts)#添加数据标签
plt.xlabel('优评区间')#设置x轴标签  
plt.ylabel('统计频次')#设置y轴标签  
plt.title('优评频数柱状图')#设置图表标题  
plt.tick_params(axis="x",labelrotation=0)
plt.grid(True)
plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
plt.show()


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


#设置最小支持度
minsup =120
start_time = time.perf_counter()
freq_itemsets_fp = fp_growth(transactions, minsup)
end_time = time.perf_counter()
for itemset in freq_itemsets_fp:
    print(f'frozenset({itemset})')
# 计算函数运行时间（结束时间 - 开始时间），并打印结果
print(f'这个函数一共运行了{end_time - start_time:.4f}秒')


#设置最小支持度
min_support=0.185
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
movies=pd.read_csv('movies.csv')
rules_with_movie_names(apriori_rules,movies)




