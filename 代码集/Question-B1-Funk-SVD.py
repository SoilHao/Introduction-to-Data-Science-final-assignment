
#第三方库
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.sparse.linalg import svds

movies= pd.read_csv('movies.csv')
rate=pd.read_csv('ratings.csv')
#使用pivot_table将数据框转换为矩阵形式  
pivot_table=rate.pivot_table(index='userId',columns='movieId',values='rating')
pivot_table.to_csv('matrix.csv',index=False)
#合并数据
data=pd.merge(rate,movies,on='movieId')

def round_half(rating):
    return np.round(rating * 2) / 2

def truncate(rating):
    if rating > 5:
        return 5
    elif rating < 0.5:
        return 0.5
    else:
        return rating

#设置隐因子数为k
k=40
#设置迭代次数为iterations
iterations=10
#设置学习率为learning_rate
learning_rate=0.01
#设置正则化率为regularization
regularization=0.01

def funk_svd(df,k,iterations,learning_rate,regularization):

   #初始化存储结果的列表
    rmse_list=[]
    mae_list=[]
    time_list=[]

   #获取用户和电影的数量
    m=len(df['userId'].unique())
    n=len(df['movieId'].unique())

   #初始化用户特征矩阵P和物品特征矩阵Q
    P=np.random.normal(scale=1./k,size=(m,k))
    Q=np.random.normal(scale=1./k,size=(n,k))

   #将用户和电影的id转换为整数索引
    df['userId']=pd.Categorical(df['userId']).codes
    df['movieId']=pd.Categorical(df['movieId']).codes

   #循环iterations次
    for i in range(iterations):
        start_time=time.time() #开始计时

       #遍历每一行数据
        for index,row in df.iterrows():
            user,movie,rating=int(row['userId']),int(row['movieId']),row['rating']
            prediction=np.dot(P[user,:],Q[movie,:]) #计算预测评分
            prediction=round_half(prediction)
            prediction=truncate(prediction)
            error=rating - prediction #计算误差

           #更新P和Q
            P[user,:] += learning_rate * (error * Q[movie,:] - regularization * P[user,:])
            Q[movie,:] += learning_rate * (error * P[user,:] - regularization * Q[movie,:])

       #计算整体的预测评分矩阵
        all_user_predicted_ratings=np.dot(P,Q.T)

       #计算RMSE
        rmse=np.sqrt(mean_squared_error(df['rating'],all_user_predicted_ratings[df['userId'],df['movieId']]))
        rmse_list.append(rmse)

       #计算MAE
        mae=mean_absolute_error(df['rating'],all_user_predicted_ratings[df['userId'],df['movieId']])
        mae_list.append(mae)

        end_time=time.time() #结束计时
        time_list.append(end_time - start_time) #计算运行时间

       #输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    return rmse_list,mae_list,time_list


#调用函数
rmse_list_funk_svd,mae_list_funk_svd,time_list=funk_svd(rate,k,iterations,learning_rate,regularization)
#输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


def bias_svd(df,k,iterations,learning_rate,regularization):

   #初始化存储结果的列表
    rmse_list=[]
    mae_list=[]
    time_list=[]

   #获取用户和电影的数量
    m=len(df['userId'].unique())
    n=len(df['movieId'].unique())

   #初始化用户特征矩阵P和物品特征矩阵Q
    P=np.random.normal(scale=1./k,size=(m,k))
    Q=np.random.normal(scale=1./k,size=(n,k))

   #初始化用户偏置向量bu和物品偏置向量bi
    bu=np.zeros(m)
    bi=np.zeros(n)

   #计算全局平均评分
    global_mean=df['rating'].mean()

   #将用户和电影的id转换为整数索引
    df['userId']=pd.Categorical(df['userId']).codes
    df['movieId']=pd.Categorical(df['movieId']).codes

   #循环iterations次
    for i in range(iterations):
        start_time=time.time() #开始计时

       #遍历每一行数据
        for index,row in df.iterrows():
            user,movie,rating=int(row['userId']),int(row['movieId']),row['rating']
            prediction=global_mean=bu[user]=bi[movie]=np.dot(P[user,:],Q[movie,:]) #计算预测评分
            prediction=round_half(prediction)
            prediction=truncate(prediction)
            error=rating - prediction #计算误差

           #更新P、Q、bu和bi
            P[user,:] += learning_rate * (error * Q[movie,:] - regularization * P[user,:])
            Q[movie,:] += learning_rate * (error * P[user,:] - regularization * Q[movie,:])
            bu[user] += learning_rate * (error - regularization * bu[user])
            bi[movie] += learning_rate * (error - regularization * bi[movie])

       #计算整体的预测评分矩阵
        all_user_predicted_ratings=global_mean=bu[:,np.newaxis]=bi[np.newaxis,:]=np.dot(P,Q.T)

       #计算RMSE
        rmse=np.sqrt(mean_squared_error(df['rating'],all_user_predicted_ratings[df['userId'],df['movieId']]))
        rmse_list.append(rmse)

       #计算MAE
        mae=mean_absolute_error(df['rating'],all_user_predicted_ratings[df['userId'],df['movieId']])
        mae_list.append(mae)

        end_time=time.time() #结束计时
        time_list.append(end_time - start_time) #计算运行时间

       #输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    return rmse_list,mae_list,time_list


#调用函数
rmse_list_bias_svd,mae_list_bias_svd,time_list=bias_svd(rate,k,iterations,learning_rate,regularization)
#输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


def svdpp(df,k,iterations,learning_rate,regularization):

   #初始化存储结果的列表
    rmse_list=[]
    mae_list=[]
    time_list=[]

   #获取用户和电影的数量
    m=len(df['userId'].unique())
    n=len(df['movieId'].unique())

   #初始化用户特征矩阵P和物品特征矩阵Q
    P=np.random.normal(scale=1./k,size=(m,k))
    Q=np.random.normal(scale=1./k,size=(n,k))

   #初始化用户偏置向量bu和物品偏置向量bi
    bu=np.zeros(m)
    bi=np.zeros(n)

   #初始化隐式反馈矩阵Y
    Y=np.random.normal(scale=1./k,size=(n,k))

   #计算全局平均评分
    global_mean=df['rating'].mean()

   #创建一个字典，存储每个用户评分过的电影
    user_rated_movies=df.groupby('userId')['movieId'].apply(list).to_dict()

   #创建一个映射，将movieId映射到一个连续的索引
    movieId_to_index={movieId: index for index,movieId in enumerate(df['movieId'].unique())}

   #循环iterations次
    for i in range(iterations):
        start_time=time.time() #开始计时

       #遍历每一行数据
        for index,row in df.iterrows():
            user,movie,rating=int(row['userId']) - 1,movieId_to_index[row['movieId']],row['rating'] #使用映射后的索引，并将userId减1
            rated_movies=[movieId_to_index[movieId] for movieId in user_rated_movies[user=1]] #使用映射后的索引，注意这里需要将userId加回1
            sqrt_N_u=np.sqrt(len(rated_movies))

           #计算隐式反馈的总和
            sum_Y=np.sum(Y[rated_movies],axis=0)

           #计算预测评分
            prediction=global_mean=bu[user]=bi[movie]=np.dot(P[user,:]=sum_Y / sqrt_N_u,Q[movie,:])
            prediction=round_half(prediction)
            prediction=truncate(prediction)
            error=rating - prediction #计算误差

           #更新P、Q、bu、bi和Y
            bu[user] += learning_rate * (error - regularization * bu[user])
            bi[movie] += learning_rate * (error - regularization * bi[movie])
            P[user,:] += learning_rate * (error * Q[movie,:] - regularization * P[user,:])
            Q[movie,:] += learning_rate * (error * (P[user,:]=sum_Y / sqrt_N_u) - regularization * Q[movie,:])
            Y[rated_movies] += learning_rate * (error * Q[movie,:] / sqrt_N_u - regularization * Y[rated_movies])

       #计算整体的预测评分矩阵
        all_user_predicted_ratings=global_mean=bu[:,np.newaxis]=bi[np.newaxis,:]=np.dot(P,Q.T)
       #将矩阵中的每个元素四舍五入到最近的0.5
        rounded_ratings=np.round(all_user_predicted_ratings * 2) / 2
       #使用clip函数将所有小于0.5的值设置为0.5，所有大于5的值设置为5
        final_ratings=np.clip(rounded_ratings,0.5,5)
       #计算RMSE
        rmse=np.sqrt(mean_squared_error(df['rating'],all_user_predicted_ratings[df['userId'] - 1,[movieId_to_index[movieId] for movieId in df['movieId']]])) #使用映射后的索引，并将userId减1
        rmse_list.append(rmse)
        
       #计算MAE
        mae=mean_absolute_error(df['rating'],all_user_predicted_ratings[df['userId'] - 1,[movieId_to_index[movieId] for movieId in df['movieId']]])
        mae_list.append(mae)

        end_time=time.time() #结束计时
        time_list.append(end_time - start_time) #计算运行时间

       #输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

   #创建一个新的DataFrame，其中行索引为实际的userId，列名为在df中有对应评分的电影标题
    titles=df['title'].unique()
    if len(titles) < n:
        titles=np.append(titles,['(UnknownMovie)'] * (n - len(titles)))
    final_ratings_df=pd.DataFrame(final_ratings,index=df['userId'].unique(),columns=titles)

    return rmse_list,mae_list,time_list,final_ratings_df


#调用函数
rmse_list_svdpp,mae_list_svdpp,time_list,all_user_predicted_ratings=svdpp(data,k,iterations,learning_rate,regularization)

#输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")


#将补全的评分矩阵保存到文件
all_user_predicted_ratings.to_csv("predicted_ratings.csv")


x=range(1,11)
y_funk_svd_rmse=rmse_list_funk_svd
y_bias_svd_rmse=rmse_list_bias_svd
y_svdpp_rmse=rmse_list_svdpp

y_funk_svd_mae=mae_list_funk_svd
y_bias_svd_mae=mae_list_bias_svd
y_svdpp_mae=mae_list_svdpp

#绘制RMSE折线图
plt.figure(figsize=(8,5))
plt.rcParams['axes.facecolor']='whitesmoke'
plt.plot(range(1,11),rmse_list_funk_svd,label='Funk-SVD')
plt.plot(range(1,11),rmse_list_bias_svd,label='BiasSVD')
plt.plot(range(1,11),rmse_list_svdpp,label='SVD++')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
#标记每一个点
plt.scatter(x,y_funk_svd_rmse,marker='*',color='blue')
plt.scatter(x,y_bias_svd_rmse,marker='D',color='orange')
plt.scatter(x,y_svdpp_rmse,marker='P',color='green')
for i in [0,4]:
    plt.annotate(f"{y_funk_svd_rmse[i]:.3f}",(x[i],y_funk_svd_rmse[i]),textcoords="offset points",xytext=(0,2),ha='center',color='blue',fontsize=9)
    plt.annotate(f"{y_bias_svd_rmse[i]:.3f}",(x[i],y_bias_svd_rmse[i]),textcoords="offset points",xytext=(0,3),ha='center',color='black',fontsize=9)
    plt.annotate(f"{y_svdpp_rmse[i]:.3f}",(x[i],y_svdpp_rmse[i]),textcoords="offset points",xytext=(0,-10),ha='center',color='green',fontsize=9)
#在背景中加入方格
plt.grid(linestyle='--',linewidth=1,alpha=0.3)
plt.legend()
plt.show()


#绘制MAE折线图
plt.figure(figsize=(8,5))
plt.rcParams['axes.facecolor']='whitesmoke'
plt.plot(range(1,11),mae_list_funk_svd,label='Funk-SVD')
plt.plot(range(1,11),mae_list_bias_svd,label='BiasSVD')
plt.plot(range(1,11),mae_list_svdpp,label='SVD++')
plt.xlabel('Iteration')
plt.ylabel('MAE')
#标记每一个点
plt.scatter(x,y_funk_svd_mae,marker='*',color='blue')
plt.scatter(x,y_bias_svd_mae,marker='D',color='orange')
plt.scatter(x,y_svdpp_mae,marker='P',color='green')
for i in [0,4]:
    plt.annotate(f"{y_funk_svd_mae[i]:.3f}",(x[i],y_funk_svd_mae[i]),textcoords="offset points",xytext=(0,2),ha='center',color='blue',fontsize=8)
    plt.annotate(f"{y_bias_svd_mae[i]:.3f}",(x[i],y_bias_svd_mae[i]),textcoords="offset points",xytext=(0,2),ha='center',color='black',fontsize=8)
    plt.annotate(f"{y_svdpp_mae[i]:.3f}",(x[i],y_svdpp_mae[i]),textcoords="offset points",xytext=(0,-10),ha='center',color='green',fontsize=8)
#在背景中加入方格
plt.grid(linestyle='--',linewidth=1,alpha=0.3)
plt.legend()
plt.show()


def recommend_movies(user_id,all_user_predicted_ratings,num_recommendations):
    
   #获取用户的预测评分
    user_predicted_ratings=all_user_predicted_ratings.loc[user_id]

   #获取评分最高的电影的名称
    recommended_movies=user_predicted_ratings.nlargest(num_recommendations).index.tolist()

    return recommended_movies


#为用户推荐电影
userId=2
num=10
recommended_movies=recommend_movies(userId,all_user_predicted_ratings,num)
print(f"对用户ID为{userId}号的用户，我们推荐以下10个电影：{'| '.join(recommended_movies)}")



def svd_func(df,k,iterations):

   #初始化存储结果的列表
    rmse_list=[]
    mae_list=[]
    time_list=[]

   #循环iterations次
    for i in range(iterations):
        start_time=time.time() #开始计时

       #创建用户-电影评分矩阵
        R_df=df.pivot(index='userId',columns='movieId',values='rating').fillna(0)

       #使用SVD进行矩阵分解
        R=R_df.values
        user_ratings_mean=np.mean(R,axis=1)
        R_demeaned=R - user_ratings_mean.reshape(-1,1)
        U,sigma,Vt=svds(R_demeaned,k=k)
        sigma=np.diag(sigma)

       #使用SVD结果进行评分预测
        all_user_predicted_ratings=np.dot(np.dot(U,sigma),Vt)=user_ratings_mean.reshape(-1,1)
        preds_df=pd.DataFrame(all_user_predicted_ratings,columns=R_df.columns)

       #计算RMSE
        rmse=np.sqrt(mean_squared_error(R,all_user_predicted_ratings))
        rmse_list.append(rmse)

       #计算MAE
        mae=mean_absolute_error(R,all_user_predicted_ratings)
        mae_list.append(mae)

        end_time=time.time() #结束计时
        time_list.append(end_time - start_time) #计算运行时间

       #输出每次迭代的结果
        print(f"第{i+1}次迭代：")
        print(f"RMSE：{rmse:.4f}")
        print(f"MAE：{mae:.4f}")
        print(f"运行时间：{end_time - start_time:.4f}秒")
        print("------------------------")

    return rmse_list,mae_list,time_list


#调用函数
rmse_list_svd,mae_list_svd,time_list=svd_func(rate,k,iterations)
#输出平均运行时间
print(f"平均运行时间：{np.mean(time_list):.4f} 秒")





