# Thanks to Siraj Raval for this module
# Refer to https://github.com/llSourcell/recommender_live for more details

import numpy as np
import pandas as pd
import math

# 基于流行度的推荐模型
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # 创建基于流行度的推荐模型
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 获取每个item的播放量，作为推荐指标
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)

        #根据播放量给歌曲排序
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    # Use the popularity based recommender system model to
    # make recommendations
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations


#基于项目的协同过滤推荐
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
    #找出给定用户听过的歌（去掉重复的）
    # Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items
    #给定歌曲，找出听过的用户
    # Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users
    #对数据集中的歌曲去重
    # Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items
    #构建相似度矩阵
    # Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):

        ####################################
        # Get users for all songs in user_songs.每首歌都找出听过的人
        ####################################
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        ###############################################
        # Initialize the item cooccurence matrix of size
        # len(user_songs) X len(songs)    矩阵大小
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        #############################################################
        # Calculate similarity between user songs and all unique songs
        # in the training data计算用户歌曲和其他所有歌曲的相似度
        #############################################################
        for i in range(0, len(all_songs)):
            # Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]    #从所有歌中确定了某首歌
            users_i = set(songs_i_data[self.user_id].unique())                            #找出了听过歌的用户

            for j in range(0, len(user_songs)):

                # Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]     #听过j的人

                # Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of listeners of songs i and j# cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                    #users_union = users_i.union(users_j)
                    for k in users_intersection:
                        #user_k_data = self.train_data[self.train_data[self.user_id] == k]
                        #user_k = list(user_k_data[self.item_id].unique())
                        user_k = self.get_user_items(k)
                        cooccurence_matrix[j, i] += 1/math.log(1 + len(user_k)*1.0)
                    cooccurence_matrix[j, i] = float(cooccurence_matrix[j, i]/math.sqrt(len(users_i)*len(users_j)))
                else:
                    cooccurence_matrix[j, i] = 0
        #print(cooccurence_matrix)
        coo_max = cooccurence_matrix.max(axis=1)
        cooccurence_matrix = cooccurence_matrix/coo_max
        #print(cooccurence_matrix)

        return cooccurence_matrix

    # Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        # Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True) #(分数e，序号i)，

        # Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 5:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    # Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # Use the item similarity based recommender system model to
    # make recommendations
    def recommend(self, user):

        ########################################
        # A. Get all unique songs for this user获取该用户听过的歌
        ########################################
        user_songs = self.get_user_items(user)

        print("No. of unique songs for the user: %d" % len(user_songs))

        ######################################################
        # B. Get all unique items (songs) in the training data获取训练集中所有歌曲
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_songs) X len(songs)构造相似度矩阵，大小为用户歌曲数×歌曲总数
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations根据相似度矩阵，topN推荐
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
        #print(df_recommendations)
        return df_recommendations

    # Get similar items to given items
    def get_similar_items(self, item_list):

        user_songs = item_list

        ######################################################
        # B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)


        return df_recommendations



