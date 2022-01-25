# iamge to sketch domain adaptation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
# from AdaboostClassifier import AdaBoostClassifier2_
# from Adaboost import AdaBoostClassifier_
from samplepoint import line 
import numpy as np
import random
import pickle
import sklearn
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

# from hellinger_distance_criterion import HellingerDistanceCriterion
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from numpy import linalg as la
import numpy as np
import pickle
import gensim.downloader as api
from gensim.models import Word2Vec
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from numpy.linalg import matrix_rank
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import matplotlib.lines as mlines

def delete_same_image (M) :
    LI=[M[0]]
    for i in range(M.shape[0]):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])                #set tmp=LI+[M[i]]
        if matrix_rank(tmp)>len(LI):    #test if M[i] is linearly independent from all (row) vectors in LI
            LI.append(M[i]) 
    zi_after=np.zeros((len(LI),M.shape[1]))
    for ii in range(zi_after.shape[0]):
        zi_after[ii]=LI[ii]          
    return zi_after


def MP (P,Phi,Rmat,y) :
    m = 1 # sample number
    N = P # eigenvector numbers in dictionary matrix N=f
    # P = 100 # feature dimension of a sample P=n2
    
    # dictionary matrix  Phi=Ti  Rmat= Z_mean
   
    Phi = Phi / np.kron(np.ones((np.size(Phi,0),1)), np.sqrt(np.sum(Phi**2,axis=0)) ) # normalize dictionary matrix

    
    # add nosie into object function for test
    # y=1
    
    x = np.zeros((N,N));  # x denote sparse coefficients
    gamma = 1.6/LA.norm(Phi)**2;
    gamma1 = 1.6/LA.norm(Rmat)**2;
    E_omp = [];
    M = 2*P
    for k in range(M):
        
        ymat = np.dot(x,Rmat)
        c = np.dot(np.transpose(Phi),(y-np.dot(Phi,ymat)) )
        E_omp = np.append(E_omp,LA.norm(y-np.dot(Phi,ymat) ) );
        i = np.argmax(np.sum(np.absolute(c), axis=1) )
        
        ymat[i,] = ymat[i,] + gamma*c[i,]
        tmp = ymat - np.dot(x,Rmat)
        x = x + gamma1*np.dot(tmp,Rmat.transpose())
        
        # projection
        I = np.nonzero(np.sum(np.absolute(x),axis=1) )
        
        x[I[0],] = np.dot(np.dot(LA.pinv(Phi[:,I[0]] ),y ),LA.pinv(Rmat) );
        
        # # record
        # if k==0:
        #     X_omp = x
        # else :
        #     X_omp = np.dstack((X_omp, x))
    
    x[np.absolute(x)<1e-5]=0 # this is the optimized sparse coefficient
    
    # plt.figure(figsize=(7,5))
    # plt.plot(np.arange(M), np.log10(E_omp), label='OMP')
    # plt.legend()
    # plt.show()
    
    return x






sketchfeature = np.load("sketchfeature2.npy")
sketchlabel = np.load("sketchlabel2.npy")
sketchfeaturetest = np.load("sketchfeaturetest.npy")
sketchlabeltest = np.load("sketchlabeltest.npy")

SK_class = np.load("dict_class_SK.npy",allow_pickle=True)
TU_class = np.load("dict_class_TU.npy",allow_pickle=True)


imagefeature = np.load("imagefeature2.npy")
imagelabel = np.load("imagelabel2.npy")
imagefeaturetest = np.load("imagefeaturetest.npy")
imagelabeltest = np.load("imagelabeltest.npy")

TUsketchfeature = np.load("TUsketchfeature.npy")
TUsketchlabel = np.load("TUsketchlabel.npy")
TUsketchfeaturetest = np.load("sketchfeature.npy")
TUsketchlabeltest = np.load("sketchlabel.npy")

TUimagefeature = np.load("TUimagefeature.npy")
TUimagelabel = np.load("TUimagelabel.npy")
TUimagefeaturetest = np.load("imagefeature.npy")
TUimagelabeltest = np.load("imagelabel.npy")



# sketch_test_class=[29,35,36,45,52,70,89,92,97,107,113,121]
sketch_test_class=list(set(sketchlabeltest))

TU_test_class=[50,57,59,75,86,123,145,148,156,173,188,200]


# for database change 

# TUsketchfeature = np.load("sketchfeature2.npy")
# TUsketchlabel = np.load("sketchlabel2.npy")
# TUsketchfeaturetest = np.load("sketchfeaturetest.npy")
# TUsketchlabeltest = np.load("sketchlabeltest.npy")

# SK_class = np.load("dict_class_SK.npy",allow_pickle=True)
# TU_class = np.load("dict_class_TU.npy",allow_pickle=True)


# TUimagefeature = np.load("imagefeature2.npy")
# TUimagelabel = np.load("imagelabel2.npy")
# TUimagefeaturetest = np.load("imagefeaturetest.npy")
# TUimagelabeltest = np.load("imagelabeltest.npy")

# sketchfeature = np.load("TUsketchfeature.npy")
# sketchlabel = np.load("TUsketchlabel.npy")
# sketchfeaturetest = np.load("sketchfeature.npy")
# sketchlabeltest = np.load("sketchlabel.npy")

# imagefeature = np.load("TUimagefeature.npy")
# imagelabel = np.load("TUimagelabel.npy")
# imagefeaturetest = np.load("imagefeature.npy")
# imagelabeltest = np.load("imagelabel.npy")



# TU_test_class=[29,35,36,45,52,70,89,92,97,107,113,121]
# sketch_test_class=[50,57,59,75,86,123,145,148,156,173,188,200]





TU_sketch=np.append(TUsketchfeature,TUsketchfeaturetest,axis=0)
TU_sketch_label=np.append(TUsketchlabel,TUsketchlabeltest,axis=0)
TU_image=np.append(TUimagefeature,TUimagefeaturetest,axis=0)
TU_image_label=np.append(TUimagelabel,TUimagelabeltest,axis=0)

for i in range (len(sketch_test_class)):
    if i==0:
        sketch_test_after=sketchfeaturetest[(sketchlabeltest==sketch_test_class[i])]
        sketch_label_after=sketchlabeltest[(sketchlabeltest==sketch_test_class[i])]
        image_test_after=imagefeaturetest[(imagelabeltest==sketch_test_class[i])]
        image_label_after=imagelabeltest[(imagelabeltest==sketch_test_class[i])]
        # TUsketch_test_after=TU_sketch[(TU_sketch_label==TU_test_class[i])]
        # TUsketch_label_after=TU_sketch_label[(TU_sketch_label==TU_test_class[i])]
        # TUimage_test_after=TU_image[(TU_image_label==TU_test_class[i])]
        # TUimage_label_after=TU_image_label[(TU_image_label==TU_test_class[i])]
    else:
        sketch_test_after=np.append(sketch_test_after,sketchfeaturetest[(sketchlabeltest==sketch_test_class[i])],axis=0)
        sketch_label_after=np.append(sketch_label_after,sketchlabeltest[(sketchlabeltest==sketch_test_class[i])],axis=0)
        image_test_after=np.append(image_test_after,imagefeaturetest[(imagelabeltest==sketch_test_class[i])],axis=0)
        image_label_after=np.append(image_label_after,imagelabeltest[(imagelabeltest==sketch_test_class[i])],axis=0)
        # TUsketch_test_after=np.append(TUsketch_test_after,TU_sketch[(TU_sketch_label==TU_test_class[i])],axis=0)
        # TUsketch_label_after=np.append(TUsketch_label_after,TU_sketch_label[(TU_sketch_label==TU_test_class[i])],axis=0)
        # TUimage_test_after=np.append(TUimage_test_after,TU_image[(TU_image_label==TU_test_class[i])],axis=0)
        # TUimage_label_after=np.append(TUimage_label_after,TU_image_label[(TU_image_label==TU_test_class[i])],axis=0)
        
        
sketch=StandardScaler(with_std=False).fit_transform(sketchfeature)
sketch_test=StandardScaler(with_std=False).fit_transform(sketch_test_after)
image=StandardScaler(with_std=False).fit_transform(imagefeature)
image_test=StandardScaler(with_std=False).fit_transform(image_test_after)
# TUimage_test=StandardScaler(with_std=False).fit_transform(TUimage_test_after)
# TUsketch_test=StandardScaler(with_std=False).fit_transform(TUsketch_test_after)

n_feature=sketch.shape[1]
n_class=len(sketch_test_class)
n_target1=1
n_target2=1
n_target=2

target_i=np.zeros((n_class,n_feature))
target_GT=np.zeros((n_class,n_feature))
dist_mean_i=np.zeros((n_class,1))
dist_mean_all_i=np.zeros((n_class,1))
dist_mean_s=np.zeros((n_class,1))
dist_mean_all_s=np.zeros((n_class,1))

transform1=[]

transform=[]
map_all=np.zeros((n_target,1))

i=5
class_index=i


str_sim = (np.expand_dims(sketch_label_after, axis=1) == np.expand_dims(image_label_after, axis=0)) * 1

distance1 = cdist(sketch_test, image_test, 'euclidean')
sim1 = 1/(1+distance1)

nq = str_sim.shape[0]
num_cores = min(multiprocessing.cpu_count(), 32)

aps1 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim1[iq]) for iq in range(nq))
map_1 = np.mean(aps1)
map_all[0]=map_1

transform2=[]

target_s=np.zeros((n_class,n_feature))

accuracy_s=np.zeros((n_target,1))
accuracy_i=np.zeros((n_target,1))




for j in range(n_target-1,n_target):
    for i in range(n_class):
        
        image_k=image_test[(image_label_after==sketch_test_class[i])]
        sketch_k=sketch_test[(sketch_label_after==sketch_test_class[i])][0:j]
        # TUsketch_k_all=TUsketch[(label2==i)]
        # TUsketch_k_normal=TUsketch_k/np.linalg.norm(TUsketch_k)
        
        # zi=np.dot(sketch_k,N)
        zi=image_k
    
        if (matrix_rank(image_k)!=image_k.shape[0]):
            image_k=delete_same_image(image_k)
        # if matrix_rank(sketch2!=sketch2.shape[0]):
        #     sketch2=delete_same_image(sketch2)
        if (matrix_rank(sketch_k)!=sketch_k.shape[0]):
            sketch_k=delete_same_image(sketch_k)
        
        
        zi_after=image_k.copy()
        ti=sketch_k.copy()
        
        css=np.dot(zi_after,zi_after.T)
        # css=np.dot(zi_after.T,zi_after)
        css=np.linalg.inv(np.linalg.cholesky(css))
        
        cst=np.dot(zi_after,ti.T)
        
        ctt=np.dot(ti,ti.T)
        # ctt=np.dot(ti.T,ti)
        ctt=np.linalg.inv(np.linalg.cholesky(ctt))
        
        udv=np.dot(css.T,cst)
        udv=np.dot(udv,ctt)
        
        L,D,RT = np.linalg.svd(udv,full_matrices=False)
        R=RT.T
        
        
        Ls=np.dot(css,L)
        Rt=np.dot(ctt,R)
        
        Ps=normalize(np.dot(zi_after.T,Ls),axis=0)
        Pt=normalize(np.dot(ti.T,Rt),axis=0)
        
        
        Q=np.dot(np.linalg.pinv(Ps),Pt)
        
    
        
        # zi_mean=np.dot(np.mean(sketch_k,axis=0),N)
        zi_mean=np.mean(image_k,axis=0)
        
        zi_norm=np.linalg.norm(zi_mean)
        # zi_mean_2=np.mean(zi_after,axis=0)
        zi_mean=normalize(zi_mean.reshape(-1, 1),axis=0).T
        zi_norm=np.linalg.norm(zi_mean)
        ti=normalize(ti,axis=1)
        
        
        y_one=np.ones((1,Pt.shape[1]))
        left=np.dot(ti,Pt)
        left=normalize(left,axis=1)
        
        
        right=np.dot(Ps,Q)
        right=normalize(right,axis=0)
        right=np.dot(zi_mean,right).T
        right=right/np.linalg.norm(right)
        
        omega=MP(Pt.shape[1],left,right,y_one.T)
        
        output=np.dot(Ps,Q)
        output=np.dot(output,omega)
        output=np.dot(output,np.linalg.pinv(Pt))
        transform1.append(output)
        
        target=np.dot(zi_mean,output)

        target=target/np.linalg.norm(target)
        target_s[i]=target
        target_i[i]=np.mean(image_k,axis=0)
        
        
        T_mean_GT=np.mean(sketch_k,axis=0)
        T_mean_GT=T_mean_GT/np.linalg.norm(T_mean_GT)
        T_mean_GT_all=np.mean(sketch_test[sketch_label_after==sketch_test_class[i]],axis=0)
        
        
        T_mean_GT_all=T_mean_GT_all/np.linalg.norm(T_mean_GT_all)
        target_GT[i]=T_mean_GT_all
        
        
        
    
        
        dist_mean_i[i]=np.linalg.norm(target_s[i]-T_mean_GT)
        dist_mean_all_i[i]=np.linalg.norm(target_s[i]-T_mean_GT_all)
        
        
        
        
        

    
    # str_sim = (np.expand_dims(sketch_label_after, axis=1) == np.expand_dims(image_label_after, axis=0)) * 1
    # str_sim = str_sim[(sketchlabeltest==list(set(imagelabeltest))[class_index])]
    
    
    distance_i = cdist(target_i,normalize(image_test), 'euclidean')
    distance_s = cdist(target_s,normalize(sketch_test), 'euclidean')
    
    retrival_image=np.argsort(distance_i,axis=1)
    retrival_image_2=np.argsort(distance_i,axis=0)[0].reshape(-1,1)
    retrival_sketch=np.argsort(distance_s,axis=0)[0].reshape(-1,1)
    
    sketch_class=np.zeros((retrival_sketch.shape))
    image_class=np.zeros((retrival_image_2.shape))
    for i in range(retrival_sketch.shape[0]):
        sketch_class[i]=sketch_test_class[retrival_sketch[i][0]]
        
    for i in range(retrival_image_2.shape[0]):
        image_class[i]=sketch_test_class[retrival_image_2[i][0]]
        
    accuracy_s=(sketch_class==sketch_label_after.reshape(-1,1)).sum()/sketch_class.shape[0]
    accuracy_i=(image_class==image_label_after.reshape(-1,1)).sum()/image_class.shape[0]



    # scale=StandardScaler(with_std=False).fit(imagefeaturetest)
    # scale=StandardScaler().fit(imagefeaturetest)
    
    # distance1 = cdist(TUsketch_test, TUimage_test, 'euclidean')
    distance2 = np.zeros((distance1.shape))
    
    for i in range(distance2.shape[0]):
        distance2[i]=distance_i[retrival_sketch[i]]
    
    # sim1 = 1/(1+distance1)
    sim2 = 1/(1+distance2)
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    
    # aps1 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim1[iq]) for iq in range(nq))
    # map_1 = np.mean(aps1)
    # map_1=0.3864
    
    aps2 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim2[iq]) for iq in range(nq))
    map_2 = np.mean(aps2)
    map_all[j]=map_2
    
    # accuracy_s=(sketch_class==sketchlabel6.reshape(-1,1)).sum()/sketch_class.shape[0]
    accuracy_class_s = np.zeros((n_class,1))
    for i in range(n_class):
        s_1=(sketch_class[sketch_label_after==sketch_test_class[i]])
        s_2=(sketch_label_after[sketch_label_after==sketch_test_class[i]]).reshape(-1,1)
        accuracy_class_s[i]=(s_1==s_2).sum()/s_1.shape[0]
        
    accuracy_class_i = np.zeros((n_class,1))
    for i in range(n_class):
        s_1=(image_class[image_label_after==sketch_test_class[i]])
        s_2=(image_label_after[image_label_after==sketch_test_class[i]]).reshape(-1,1)
        accuracy_class_i[i]=(s_1==s_2).sum()/s_1.shape[0]
    
    print("target number:", j, "   map:",map_2,"   accuracy_s:",accuracy_s)

plt.figure(figsize=(7,5))
plt.plot(range(n_target), map_all)
plt.legend()
plt.xlabel('training samples per class')
plt.ylabel('map@all')
plt.show()




# # ts = TSNE(n_components=2, init='pca', random_state=0)
# # result = ts.fit_transform(target_i)

# result = np.append(normalize(sketch_test),target_s,axis=0)
# result = np.append(result,target_GT,axis=0)
# # result = ts.fit_transform(result)

# # sns.set(rc={'figure.figsize':(11.7,8.27)})
# # palette = sns.color_palette("Spectral", 24)
# # sns.scatterplot(result[:,0], result[:,1], hue=np.append(sketch_label_after*10,sketch_test_class,axis=0),legend = False,palette=palette)
# # g._remove_legend()
# # fig = plot_embedding(result, TUsketch_label_after, 't-SNE Embedding of digits')
# # plt.show()

# ts = TSNE(n_components=3, init='pca', random_state=0)
# # result = ts.fit_transform(target_i)
# ts_result = ts.fit_transform(result)

# class_index_plot=np.argmin(accuracy_class_s)
# # class_index_plot=np.argmax(accuracy_class_s)

# # class_index_plot=np.argmin(dist_mean_all_i)
# # class_index_plot=np.argmax(dist_mean_all_i)

# result=ts_result[0:ts_result.shape[0]-target_s.shape[0]*2][sketch_label_after==sketch_test_class[class_index_plot]]
# fig = plt.figure(dpi=600)

# result1=result[0:n_target]
# result2=result[n_target:]

# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(result1[:,0], result1[:,1], result1[:,2], c= 'c', s = 50, marker='.')
# ax.scatter(result2[:,0], result2[:,1], result2[:,2], c= 'r', s = 1, marker='.')

# index_mean_ts=ts_result.shape[0]-(target_s.shape[0]-class_index_plot+target_GT.shape[0])
# index_mean_GT=ts_result.shape[0]-(target_s.shape[0]-class_index_plot)
# ax.scatter(ts_result[index_mean_ts,0], ts_result[index_mean_ts,1], ts_result[index_mean_ts,2], c= 'b', s = 300, marker='.')
# ax.scatter(ts_result[index_mean_GT,0], ts_result[index_mean_GT,1], ts_result[index_mean_GT,2], c= 'g', s = 300, marker='.')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title("t-SNE")
# plt.axis('tight')

# yellow_dot = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
#                           markersize=10, label='other sample feature alignment')
# red_dot = mlines.Line2D([], [], color='c', marker='.', linestyle='None',
#                           markersize=10, label='few-shot sample feature alignment')
# blue_dot = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
#                           markersize=10, label='estimated mean')
# green_dot = mlines.Line2D([], [], color='green', marker='.', linestyle='None',
#                           markersize=10, label='Ground Truth mean')

# plt.legend(handles=[red_dot,yellow_dot,blue_dot, green_dot],loc='upper right',fontsize='xx-small')


# plt.show()