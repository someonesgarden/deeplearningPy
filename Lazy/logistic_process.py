#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd


# N = 100
# D = 2
# X = np.random.randn(N,D)   #(N,D)
# ones = np.array([[1]*N]).T   #(N,1)
# Xb = np.concatenate((ones, X), axis=1)  #(N,D+1)
# w = np.random.randn(D+1)  #(D+1,1)
# z = Xb.dot(w)  #(N,D+1) x (D+1,1) = (N,1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def get_data():

    #Pandasを利用してcsvを解釈
    df = pd.read_csv("data/ecommerce_data.csv")
    data = df.as_matrix()
    X = data[:, :-1]  #最後の行以外
    Y = data[:, -1]  #最後の行

    #D：最終行 user_action
    #D-1:

    #Normalize with Z value
    X[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()  #n_products_viewed
    X[:, 2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()  #visit_duration

    #Categorical Value of "time_of_day"
    N, D = X.shape

    X2 = np.zeros((N, D-1 + 4)) # ４つの異なったカテゴリーカルバビューがあるため増やす
    X2[:,0:(D-1)] = X[:,0:(D-1)] #カテゴリカルバリュー以外のところは同じなのでそのまま入れる



    # ONE HOTのやり方１：forループを使ってカテゴリカルバリューをOne Hotに変える方法
    for n in range(N):  #全行
        t = int(X[n,D-1]) #カテゴリカルバリューをインデックスに変換(t=0,1,2,3)
        X2[n, t+D-1] = 1

    # ONE HOTのやり方２：numpyの配列計算を使ってカテゴリカルバリューをOne Hotに変える方法
    Z = np.zeros((N, 4)) #カテゴリカルの数だけからの配列を用意
    #後ろから二番目のtime_of_day（D-1)に(0,1,2,3)のどれかが入っているので、
    #その値からOne Hot Valueを作成する
    Z[np.arange(N),X[:,D-1].astype(np.int32)] = 1
    X2[:, -4:] = Z   # X2の後ろ四つにZを置き換える

    assert(np.abs(X2[:,-4:]-Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2  = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

#
X2, Y2 = get_binary_data()
