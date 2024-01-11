
import pandas as pd
import numpy as np
import time
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

#加载数据
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

#合并数据
data = pd.merge(ratings,movies,on='movieId')


def hybrid_cf(data,user_id):
    start_time = time.time()
   #创建用户-电影评分矩阵
    matrix = data.pivot_table(index='userId',columns='title',values='rating')
    matrix = matrix.fillna(0)
   #计算用户和物品之间的相似度
    user_similarity = cosine_similarity(matrix.fillna(0))
    item_similarity = cosine_similarity(matrix.T.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity,index=matrix.index,columns=matrix.index)
    item_similarity_df = pd.DataFrame(item_similarity,index=matrix.columns,columns=matrix.columns)
   #矩阵分解
    U,sigma,Vt = svds(matrix.values,k=50)
    sigma = np.diag(sigma)
   #预测评分
    predicted_ratings = np.dot(np.dot(U,sigma),Vt)
    predicted_df = pd.DataFrame(predicted_ratings,index=matrix.index,columns=matrix.columns)
    user_ratings = matrix.loc[user_id]
    prediction = predicted_df.loc[user_id]
    actual = []
    for title,rating in user_ratings.items():
        if not pd.isnull(rating):
            actual.append(rating)
   #考虑用户和物品的相似度
    similar_users = user_similarity_df.loc[user_id]
    similar_items = item_similarity_df.loc[prediction.index].mean()
    prediction = prediction * similar_users * similar_items

   #限制预测评分在0.5到5之间，并且只能取0.5,1,1.5,2,2.5,3,3.5,4,4.5,5
    prediction = np.round(prediction * 2) / 2
    prediction = prediction.clip(0.5,5)

   #如果所有预测评分都被过滤掉，那么使用平均评分作为预测评分
    if len(prediction) == 0:
        prediction = [matrix.values.mean()] * len(actual)

   #计算RMSE和MAE
    prediction_values = list(prediction) if isinstance(prediction,list) else prediction.tolist()
    prediction_values = [value if not np.isnan(value) and not np.isinf(value) else 0 for value in prediction_values]
    prediction_values = prediction_values[:len(actual)]
    rmse = np.sqrt(mean_squared_error(actual,prediction_values))
    mae = mean_absolute_error(actual,prediction_values)

   #计算运行时间
    run_time = time.time() - start_time
    print(f"RMSE：{rmse:.4f}\nMAE：{mae:.4f}\n运行时间：{run_time:.4f}秒")
    return prediction


def recommend_movies(data,user_id):
   #读取预测评分矩阵
    predicted_df = pd.read_csv('predicted_ratings.csv',index_col=0)
   #获取用户的预测评分
    user_ratings = predicted_df.loc[user_id]
   #排序并获取前10部电影
    top_movies = user_ratings.sort_values(ascending=False).index[:10]
   #格式化输出
    print(f"对用户ID为{user_id}的用户，我们推荐以下10个电影：{'|'.join(top_movies)}")

user_id = 3
prediction =hybrid_cf(data,user_id)

#输出对指定用户的电影推荐
recommend_movies(data,user_id)




