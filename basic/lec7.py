# import numpy as np
# import pandas as pd
# from pandas import Series
# import matplotlib.pyplot as plt
#
#
# np.zeros(4)
# np.ones(3)
# np.empty(5)
# np.eye(4)
# np.arange(3, 20, 2)
#
# points = np.arange(-5, 5, 0.01)
#
# dx, dy = np.meshgrid(points, points)
#
# plt.imshow(dx)
# plt.show()
#
# plt.imshow(dy)
# plt.colorbar()
# plt.show()
#
# A = np.array([1,2,3,4])
# B = np.array([12,23,34,45])
# condition = np.array([True, False, False, True])
#
# answer = [a if cond else b for a,b,cond in zip(A,B,condition)] # 結果はリスト！
# answer2 = np.where(condition, A, B) # 結果はnp.array
#
# from numpy.random import randn
#
# arr = randn
#
# np.unique
# np.in1d() #
#
#
# np.save
# np.savaz('ziparray.npz', x=arr1, y=arr2)
# np.load('')
#
# np.savetxt('')
# np.loadtxt('')
#
# # numpy.arrayとpandas.Seriesの違いは、
# # Seriesはindexという名前をつけることができる！！！
#
# obj = Series([1,2,2,3])
# obj.value => numpyのarray!!!
#
# obj.index
#
# #pandas SEries
# obj.to_dict()
# pd.is_null(obj)
# pd.notnull(obj)
#
#
# # DataFrame
#
# df = pd.read_clipboard()  # コピペで選んだデータをそのままDataFrameに入れることができる!!!