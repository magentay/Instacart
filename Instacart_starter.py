import pandas as pd
import numpy as np
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor


datadir = '/data/Instacart/'
resultdir = '/data/Instacart/results/'

orders = pd.read_csv(datadir + 'orders.csv')

train = pd.read_csv(datadir + 'train.csv')

gth = pd.read_csv(datadir +"order_products__train.csv" )
gth = pd.merge(gth, orders[['order_id','user_id']] , on = 'order_id', how='left')
gth['y'] = 1

train = pd.merge(train, gth[['user_id', 'product_id','y']], on =['user_id', 'product_id'], how = 'left')
train.fillna(0, inplace = True)

feats = ['weights_sum', 'weights_mean',
       'order_weight_max', 'order_weight_count', 'order_weight_sum',
       'time_weight_max', 'time_weight_sum', 'days_sum', 'days_max',
       'days_min', 'days_count', 'mean_gap', 'weights',
       'product_user_reorder_ratio', 'product_reorder_ratio',
       'product_user_ratio', 'aisle_reorder_ratio', 'dept_reorder_ratio']

gc.collect()

print("running random forest..........")
rf =   RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1)
rf.fit(train[feats], train.y)
train['y_rf'] = rf.predict(train[feats])
gc.collect()

print("running xgboost..........")
model = XGBRegressor()
model.fit(train[feats], train.y)
train['y_xgb'] = model.predict(train[feats])

gc.collect()

def getProduct(row):
    l = int(np.ceil(row['average_product_per_order']))
    return ' '.join(  [str(x) for x in row['product_id'][:l] ])

def predict_product(train, trainset, outfile):

    train['y_pred'] = train['y_rf']+ train['y_xgb']
    train.sort_values(by=['user_id','y_pred'], ascending = False, inplace=True)
    myfun = lambda x : list(x)
    predict = train.groupby('user_id').agg({'product_id':myfun, "average_product_per_order":'mean'}).reset_index()

    predict['products'] = predict.apply(getProduct, axis=1)


    predict = pd.merge(trainset, predict, on ='user_id', how = 'inner')

    predict[['order_id', 'products']].to_csv( resultdir + outfile, index = False)



trainset = orders.ix[orders.eval_set == 'train']
predict_product(train, trainset, 'train_0001.csv')
del train

testset = orders.ix[orders.eval_set == 'test']
test = pd.read_csv(datadir + 'test.csv')
test.fillna(0, inplace=True)
test['y_rf'] = rf.predict(test[feats])
test['y_xgb'] = model.predict(test[feats])

predict_product(test, testset, 'test_0001.csv')


gc.collect()

