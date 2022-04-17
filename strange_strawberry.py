from keras.preprocessing.image import ImageDataGenerator
path = 'data//labeled_data'
dst_path = 'data//gen_data' #生成图片地方
datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.02,horizontal_flip=True,vertical_flip=True) #创建图片生成器
gen = datagen.flow_from_directory(path,target_size=(224,224),batch_size=2,save_to_dir=dst_path,save_prefix='gen',save_format='jpg') #创建读取图片的方法 batch_size每一次生成的图片数量;save_prefix存储图片的前缀
for i in range(100):
    gen.next()

#图片加载
from keras.preprocessing.image import load_img,img_to_array
img_path = '1.jpg'
img = load_img(img_path,target_size=(224,224))
#print(img)

#可视化
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.imshow(img)

img = img_to_array(img) #格式转化
#print(type(img))
#print(img.shape)

from keras.applications.vgg16 import preprocess_input
import numpy  as np
img_p = np.expand_dims(img,axis=0)  #数据维度转化与预处理
#print(img_p.shape)
img_p = preprocess_input(img_p)

#VGG16特征提取
from keras.applications.vgg16 import VGG16
model_vgg16 = VGG16(weights='imagenet',include_top=False) #include_top不需要输出层
features = model_vgg16.predict(img_p)
#print(features.shape,features)

features = features.reshape(1,7*7*512) #特征展开flatten
#print(features.shape)

import os
folder = 'data//training_data'
files_name = os.listdir(folder)  #批量图片的路径获取
#print(files_name)
img_path = [] #用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder+'//'+i for i in img_path]
#print(img_path)

#VGG16特征提取的方法定义
def modelProcess(img_path,model):
    img = load_img(img_path,target_size=(224,224))
    img = img_to_array(img)
    img_p = np.expand_dims(img,axis=0) #添加上一个维度
    img_p = preprocess_input(img_p)
    img_vgg = model.predict(img_p)
    img_vgg = img_vgg.reshape(1,7*7*512)
    return img_vgg

#批量提取图片特征
features_train = np.zeros([len(img_path),7*7*512])
#print(features_train)
for i in range(len(img_path)):
    features_temp = modelProcess(img_path[i],model_vgg16)
    features_train[i] = features_temp
    print('preprocess:',img_path[i])

#样本数量与特征数
#print(features_train.shape)

#X赋值
X = features_train
#print(X.shape)

#kmeans模型聚类分析
from sklearn.cluster import KMeans
vgg_kmeans = KMeans(n_clusters=2,max_iter=3000)
#训练
vgg_kmeans.fit(X)

#预测
y_predict_km = vgg_kmeans.predict(X)
print(y_predict_km)

#预测结果分布统计
import pandas as pd
print(pd.value_counts(y_predict_km))

#普通草莓id
normal_strawberry_id = 1

fig2 = plt.figure(figsize=(10,40))
for i in range(48):
    for j in range(5):
        img = load_img(img_path[i*5+j])
        plt.subplot(48,5,i*5+j+1)
        plt.title('n-strawberry' if y_predict_km[i*5+j]==normal_strawberry_id else 's-strawberry')
        plt.imshow(img)
        plt.axis('off')

#批量图片的路径获取
import os
folder = 'task2_data//test_data'
files_name = os.listdir(folder)
# print(files_name)
img_path = []#用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]
print(img_path)

#批量提取图片特征
features_test = np.zeros([len(img_path),7*7*512])
for i in range(len(img_path)):
    features_temp = modelProcess(img_path[i],model_vgg16)
    features_test[i] = features_temp
    print('preprocess:',img_path[i])

#X_test
X_test = features_test
print(X_test.shape)

#测试数据预测
y_predict_km_test = vgg_kmeans.predict(X_test)
print(y_predict_km_test)

fig2 = plt.figure(figsize=(10,10))
for i in range(3):
    for j in range(4):
        img = load_img(img_path[i*4+j])
        plt.subplot(3,4,i*4+j+1)
        plt.title('n-strawberry' if y_predict_km_test[i*4+j]==normal_strawberry_id else 's-strawberry')
        plt.imshow(img)
        plt.axis('off')


#meanshift模型替代kmeans模型
from sklearn.cluster import MeanShift, estimate_bandwidth
#获取合适的meanshift半径
bw = estimate_bandwidth(X,n_samples=150)
print(bw)

vgg_ms = MeanShift(bandwidth=bw)
#模型训练
vgg_ms.fit(X)

#预测
y_predict_ms = vgg_ms.predict(X)
print(y_predict_ms)

#预测结果分布统计
print(pd.value_counts(y_predict_ms))

#普通草莓id
normal_strawberry_id = 0

#批量图片的路径获取
import os
folder = 'task2_data//training_data'
files_name = os.listdir(folder)
# print(files_name)
img_path = []#用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]
print(img_path)

#批量图片的路径获取
import os
folder = 'task2_data//test_data'
files_name = os.listdir(folder)
# print(files_name)
img_path = []#用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]
print(img_path)

#测试数据预测
y_predict_ms_test = vgg_ms.predict(X_test)
print(y_predict_ms_test)

fig2 = plt.figure(figsize=(10,10))
for i in range(3):
    for j in range(4):
        img = load_img(img_path[i*4+j])
        plt.subplot(3,4,i*4+j+1)
        plt.title('n-strawberry' if y_predict_ms_test[i*4+j]==normal_strawberry_id else 's-strawberry')
        plt.imshow(img)
        plt.axis('off')

#PCA主成分分析
from sklearn.preprocessing import StandardScaler
standards = StandardScaler()
X_standard = standards.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=200)
X_pca = pca.fit_transform(X_standard)

#维度确认
#print(X.shape,X_pca.shape)

#计算降维后的方差比例
var_ratio = pca.explained_variance_ratio_
#print(np.sum(var_ratio))

#创建第二个ms模型
#获取ms的半径
bw_2 = estimate_bandwidth(X_pca,n_samples=150)
#print(bw_2)

vgg_pca_ms = MeanShift(bandwidth=bw_2)
vgg_pca_ms.fit(X_pca)

#模型预测
y_predict_pca_ms = vgg_pca_ms.predict(X_pca)
#print(y_predict_pca_ms)

print(pd.value_counts(y_predict_pca_ms))

#普通草莓id
normal_strawberry_id = 0

#批量图片的路径获取
import os
folder = 'task2_data//training_data'
files_name = os.listdir(folder)
# print(files_name)
img_path = []#用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]
print(img_path)

fig3 = plt.figure(figsize=(10,40))
for i in range(48):
    for j in range(5):
        img = load_img(img_path[i*5+j])
        plt.subplot(48,5,i*5+j+1)
        plt.title('n-strawberry' if y_predict_pca_ms[i*5+j]==normal_strawberry_id else 's-strawberry')
        plt.imshow(img)
        plt.axis('off')


#批量图片的路径获取
import os
folder = 'task2_data//test_data'
files_name = os.listdir(folder)
# print(files_name)
img_path = []#用于存储图片路径
for i in files_name:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path = [folder + '//' + i for i in img_path]
print(img_path)

X_test_standard = standards.transform(X_test)
#测试数据pca降维
X_test_pca = pca.transform(X_test_standard)
print(X_test.shape,X_test_pca.shape)

#测试数据预测
y_predict_pca_ms_test = vgg_pca_ms.predict(X_test_pca)
print(y_predict_pca_ms_test)

fig2 = plt.figure(figsize=(10,10))
for i in range(3):
    for j in range(4):
        img = load_img(img_path[i*4+j])
        plt.subplot(3,4,i*4+j+1)
        plt.title('n-strawberry' if y_predict_pca_ms_test[i*4+j]==normal_strawberry_id else 's-strawberry')
        plt.imshow(img)
        plt.axis('off')