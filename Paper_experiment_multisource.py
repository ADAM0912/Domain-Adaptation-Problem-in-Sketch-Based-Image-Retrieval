# delete useless dataset


from __future__ import division
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
# from AdaboostClassifier import AdaBoostClassifier2_
# from Adaboost import AdaBoostClassifier_
from samplepoint import line 
import numpy as np
import random
import pickle
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from numpy.linalg import matrix_rank

# from hellinger_distance_criterion import HellingerDistanceCriterion
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines

import scipy as scp
import pylab as pyl

# import DICTOLpy
from DICTOLpy import utils
from DICTOLpy import ODL





import scipy as scp
import pylab as pyl


from numpy import linalg as LA
from nt_toolbox.general import *
from nt_toolbox.signal import *

import warnings
warnings.filterwarnings('ignore')

import os
from numpy import *

from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

def MP (P,Phi,Rmat,y):
    
    m = 1 # sample number
    N = P # eigenvector numbers in dictionary matrix N=f
    # P = 100 # feature dimension of a sample P=n2
    
    Phi = Phi / np.kron(np.ones((np.size(Phi,0),1)), np.sqrt(np.sum(Phi**2,axis=0)) ) # normalize dictionary matrix
    # Rmat = np.random.rand(N,m)
    
    # add nosie into object function for test
    

    # sigma = 0.05 * LA.norm(np.dot(np.dot(Phi,x0),Rmat ) )/np.sqrt(P)
    # y = np.ones((P,1))   # y denotes object function
    
    x = np.zeros((N,N));  # x denote sparse coefficients
    gamma = 1.6/LA.norm(Phi)**2;
    gamma1 = 1.6/LA.norm(Rmat)**2;
    E_omp = [];
    # M = 2*P
    M=20
    xiter = 25;
    tau = 1/LA.norm(np.dot(Rmat,np.transpose(Rmat) ) )
    for k in range(M):
        
        ymat = np.dot(x,Rmat)
        c = np.dot(np.transpose(Phi),(y-np.dot(Phi,ymat)) )
        E_omp = np.append(E_omp,LA.norm(y-np.dot(Phi,ymat) ) );
        i = np.argmax(np.sum(np.absolute(c), axis=1) )
        
        ymat[i,] = ymat[i,] + gamma*c[i,]
        tmp = ymat - np.dot(x,Rmat)
        x = x + gamma1*np.dot(tmp,Rmat.transpose())
        
        # projection
        I = np.nonzero(np.sum(np.absolute(c),axis=1) )
        
        ymat[I[0],] = np.dot(LA.pinv(Phi[:,I[0]] ),y );
        
        for ii in range(xiter):
            x = x + tau * np.dot(ymat - np.dot(x,Rmat ), Rmat.transpose() );
            x = x / np.kron(np.ones((np.size(x,0),1)), np.sqrt(np.sum(x**2,axis=0)) )
        
        # record
        if k==0:
            X_omp = x
        else :
            X_omp = np.dstack((X_omp, x))
    
    x[np.absolute(x)<1e-5]=0 
    
    # plt.figure(figsize=(7,5))
    # plt.plot(np.arange(M), np.log10(E_omp), label='OMP')
    # plt.legend()
    # plt.show()
    
    return x


def CCA (zi,ti):
        css=np.dot(zi,zi.T)
        css=np.linalg.inv(np.linalg.cholesky(css))
        
        cst=np.dot(zi,ti.T)
        
        ctt=np.dot(ti,ti.T)
        ctt=np.linalg.inv(np.linalg.cholesky(ctt))
        
        udv=np.dot(css.T,cst)
        udv=np.dot(udv,ctt)
        
        L,D,RT = np.linalg.svd(udv,full_matrices=True)
        R=RT.T
        
        # L=L[:,0:D.shape[0]]
        
        Ls=np.dot(css,L)
        Rt=np.dot(ctt,R)
        

        Ps=normalize(np.dot(zi.T,Ls),axis=0)
        Pt=normalize(np.dot(ti.T,Rt),axis=0)
        
        Q=np.dot(np.linalg.pinv(Ps),Pt)
        
        return Ps,Pt,Q
    
def onlinelearning(A_our, Y_m, p, eta_x, thr, C, eta_A, tol_A):
   AtY = np.dot(A_our.transpose(),Y_m )
   XS = AtY*(np.absolute(AtY) >= C/2 )
   AtA_our = np.dot(A_our.transpose(),A_our )
   err_int_A = LA.norm(np.dot(A_our,XS ) - Y_m,'fro')/LA.norm(Y_m,'fro')
   change_A = 1
   while change_A > tol_A:
       XS = XS - eta_x*(np.dot(AtA_our,XS ) - np.dot(A_our.transpose(),Y_m ) );
       XS = XS * (np.absolute(XS) > thr)
       err_int_A = np.append(err_int_A, LA.norm(np.dot(A_our,XS ) - Y_m,'fro')/LA.norm(Y_m,'fro') );
       change_A = np.absolute(err_int_A[-1] - err_int_A[-2] )
   gr = np.dot((Y_m - np.dot(A_our,XS) ), np.transpose(np.sign(XS) ) ) / p
   A_our = A_our + (eta_A)*gr
   A_our = normalize(A_our, axis=0, norm='l2')
   return A_our, XS;

def OrthogonalMatchingPursuit(y, Phi):
    N = np.size(Phi,1)
    M = 2*N;
    x = np.zeros(N).transpose();

    for k in range(M):
    # E_omp(k) = norm(y-Phi*x);
        corre = np.dot(Phi.transpose(),(y - np.dot(Phi,x )) );
        i = np.argmax(np.absolute(corre))
        x[i] = x[i] + corre[i];
        
        I = np.nonzero(np.absolute(x) )
        x[I[0]] = np.dot(LA.pinv(Phi[:,I[0]] ),y );

    x = x*(np.absolute(x)>1.e-6)
    return x
    
    
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


# sketchfeature3 = np.load("./feature_extracted/DM_real_feature_small.npy")
# sketchlabel3 = np.load("./feature_extracted/DM_real_label_small.npy")

# sketchfeature5 = np.load("./feature_extracted/DM_sketch_feature_small.npy")
# sketchlabel5 = np.load("./feature_extracted/DM_sketch_label_small.npy")

# sketchfeature6 = np.load("./feature_extracted/DM_clipart_feature_small.npy")
# sketchlabel6 = np.load("./feature_extracted/DM_clipart_label_small.npy")

# sketchfeature1 = np.load("./feature_extracted/DM_quickdraw_feature_small.npy")
# sketchlabel1 = np.load("./feature_extracted/DM_quickdraw_label_small.npy")

# sketchfeature4 = np.load("./feature_extracted/DM_painting_feature_small.npy")
# sketchlabel4 = np.load("./feature_extracted/DM_painting_label_small.npy")


# sketchfeature2 = np.load("./feature_extracted/DM_infograph_feature_small.npy")
# sketchlabel2= np.load("./feature_extracted/DM_infograph_label_small.npy")



sketchfeature1 = np.load("./feature_extracted/DM_real_feature.npy")
sketchlabel1 = np.load("./feature_extracted/DM_real_label.npy")

sketchfeature3 = np.load("./feature_extracted/DM_sketch_feature.npy")
sketchlabel3 = np.load("./feature_extracted/DM_sketch_label.npy")

sketchfeature6 = np.load("./feature_extracted/DM_clipart_feature.npy")
sketchlabel6 = np.load("./feature_extracted/DM_clipart_label.npy")

sketchfeature4 = np.load("./feature_extracted/DM_quickdraw_feature.npy")
sketchlabel4 = np.load("./feature_extracted/DM_quickdraw_label.npy")

sketchfeature2 = np.load("./feature_extracted/DM_painting_feature.npy")
sketchlabel2 = np.load("./feature_extracted/DM_painting_label.npy")


sketchfeature5 = np.load("./feature_extracted/DM_infograph_feature.npy")
sketchlabel5= np.load("./feature_extracted/DM_infograph_label.npy")




# X_train1, X_test1, y_train1, y_test1 = train_test_split(sketchfeature6,sketchlabel6,random_state=10)

# sketchfeature6 = X_train1
# sketchlabel6 = y_train1
# sketchfeature6_test=X_test1
# sketchlabel6_test=y_test1

class_list=list(set(sketchlabel6))

sample_per_class= 300

n_feature=sketchfeature1.shape[1]

n_class=len(class_list)

n_target=4

n_dataset=5

d=30

dist_mean=np.zeros((n_class,1))
dist_mean_all=np.zeros((n_class,1))

delete_index_class=np.zeros((n_class,1))

Average_dist_mean=np.zeros((n_target,1))
Average_dist_mean_all=np.zeros((n_target,1))

class_mean=[]



# WITHOUT PCA

# X_train1, X_test1, y_train1, y_test1 = train_test_split(sketch,label1)



# X_train2, X_test2, y_train2, y_test2 = train_test_split(TUsketch,label2)


# sketchfeature1=sketchfeature1-np.mean(sketchfeature1,axis=0)
# sketchfeature2=sketchfeature2-np.mean(sketchfeature2,axis=0)
# sketchfeature3=sketchfeature3-np.mean(sketchfeature3,axis=0)
# sketchfeature4=sketchfeature4-np.mean(sketchfeature4,axis=0)
# sketchfeature5=sketchfeature5-np.mean(sketchfeature5,axis=0)
# sketchfeature6=sketchfeature6-np.mean(sketchfeature6,axis=0)

sketchfeature1=StandardScaler(with_std=False).fit_transform(sketchfeature1)
sketchfeature2=StandardScaler(with_std=False).fit_transform(sketchfeature2)
sketchfeature3=StandardScaler(with_std=False).fit_transform(sketchfeature3)
sketchfeature4=StandardScaler(with_std=False).fit_transform(sketchfeature4)
sketchfeature5=StandardScaler(with_std=False).fit_transform(sketchfeature5)
sketchfeature6=StandardScaler(with_std=False).fit_transform(sketchfeature6)




str_sim = (np.expand_dims(sketchlabel6, axis=1) == np.expand_dims(sketchlabel1, axis=0)) * 1    
map_all=np.zeros((n_target,1))

var=np.zeros((n_class,n_dataset))



for j in range(n_target-1,n_target):
    sketchfeature6_small=np.zeros((1*(j+1)*n_class,n_feature))
    sketchlabel6_small=np.zeros((1*(j+1)*n_class,))
    targetnumber=1*(j+1)
    n2=targetnumber
    for k in range(n_class):
        sketchfeature6_i=sketchfeature6[sketchlabel6==(list(set(sketchlabel6))[k])]
        sketchlabel6_i=sketchlabel6[sketchlabel6==(list(set(sketchlabel6))[k])]
        sketchfeature6_small[targetnumber*k:targetnumber*k+targetnumber]=sketchfeature6_i[0:targetnumber]
        sketchlabel6_small[targetnumber*k:targetnumber*k+targetnumber]=sketchlabel6_i[0:targetnumber]
        

    
    target_i=np.zeros((n_class,n_feature))
    target_s=np.zeros((n_class,n_feature))
    target_GT_i=np.zeros((n_class,n_feature))
    target_GT_s=np.zeros((n_class,n_feature))
    
    for i in range(n_class):
        
        
        
        # sketch1=sketchfeature1[sketchlabel1==(list(set(sketchlabel1))[i])]
        # sketch2=sketchfeature2[sketchlabel2==(list(set(sketchlabel2))[i])]
        # sketch3=sketchfeature3[sketchlabel3==(list(set(sketchlabel3))[i])]
        # sketch4=sketchfeature4[sketchlabel4==(list(set(sketchlabel4))[i])]
        # sketch5=sketchfeature5[sketchlabel5==(list(set(sketchlabel5))[i])]
        # sketch6=sketchfeature6_small[sketchlabel6_small==(list(set(sketchlabel6_small))[i])]
        # sketch6_all=sketchfeature6[sketchlabel6==(list(set(sketchlabel6))[i])]
        
        sketch=[]
        sketch.append(class_list)
        sketch.append(sketchfeature1[sketchlabel1==(list(set(sketchlabel1))[i])])
        sketch.append(sketchfeature2[sketchlabel2==(list(set(sketchlabel2))[i])])
        sketch.append(sketchfeature3[sketchlabel3==(list(set(sketchlabel3))[i])])
        sketch.append(sketchfeature4[sketchlabel4==(list(set(sketchlabel4))[i])])
        sketch.append(sketchfeature5[sketchlabel5==(list(set(sketchlabel5))[i])])
        sketch.append(sketchfeature6_small[sketchlabel6_small==(list(set(sketchlabel6_small))[i])])
        
        if matrix_rank(sketch[1]!=sketch[1].shape[0]):
            sketch[1]=delete_same_image(sketch[1])
        if matrix_rank(sketch[2]!=sketch[2].shape[0]):
            sketch[2]=delete_same_image(sketch[2])
        if matrix_rank(sketch[3]!=sketch[3].shape[0]):
            sketch[3]=delete_same_image(sketch[3])
        if matrix_rank(sketch[4]!=sketch[4].shape[0]):
            sketch[4]=delete_same_image(sketch[4])
        if matrix_rank(sketch[5]!=sketch[5].shape[0]):
            sketch[5]=delete_same_image(sketch[5])
        if matrix_rank(sketch[6]!=sketch[6].shape[0]):
            sketch[6]=delete_same_image(sketch[6])
        
        n2=sketch[6].shape[0]
        # var=np.zeros((n_dataset,1))
        dataset_index=[1,2,3,4,5]
        
        # var[i][0]=np.mean(cdist(normalize(np.mean(sketch[1],axis=0).reshape(1,-1)),normalize(sketch[1]), 'euclidean'))
        # var[i][1]=np.mean(cdist(normalize(np.mean(sketch[2],axis=0).reshape(1,-1)),normalize(sketch[2]), 'euclidean'))
        # var[i][2]=np.mean(cdist(normalize(np.mean(sketch[3],axis=0).reshape(1,-1)),normalize(sketch[3]), 'euclidean'))
        # var[i][3]=np.mean(cdist(normalize(np.mean(sketch[4],axis=0).reshape(1,-1)),normalize(sketch[4]), 'euclidean'))
        # var[i][4]=np.mean(cdist(normalize(np.mean(sketch[5],axis=0).reshape(1,-1)),normalize(sketch[5]), 'euclidean'))
        # delete_index=np.argmax(var[i])
        # delete_index_class[i]=delete_index+1
        # dataset_index.remove(delete_index+1)
        
        n_dataset_class=len(dataset_index)

        
        zi=[]
        A_all=[]
        for ii in range (n_dataset_class):
            index=dataset_index[ii]
            z=np.mean(sketch[index],axis=0)
            zi_1=z/np.linalg.norm(z)
            
            # zi_2=np.mean(sketch2,axis=0).reshape(1, -1)  
            # zi_3=np.mean(sketch3,axis=0).reshape(1, -1)  
            # zi_4=np.mean(sketch4,axis=0).reshape(1, -1)  
            # zi_5=np.mean(sketch5,axis=0).reshape(1, -1)  
        
            # zi_1=zi_1/np.linalg.norm(zi_1)
            # zi_2=zi_2/np.linalg.norm(zi_2)
            # zi_3=zi_3/np.linalg.norm(zi_3)
            # zi_4=zi_4/np.linalg.norm(zi_4)
            # zi_5=zi_5/np.linalg.norm(zi_5)
    
            Ps_1,Pt_1,Q_1=CCA(sketch[index],sketch[6])
            # Ps_2,Pt2,Q2=CCA(sketch2,sketch6)
            # Ps3,Pt3,Q3=CCA(sketch3,sketch6)
            # Ps4,Pt4,Q4=CCA(sketch4,sketch6)
            # Ps5,Pt5,Q5=CCA(sketch5,sketch6)
            
            
            
            s2_1=np.linalg.norm(zi_1)
            # s2_2=np.linalg.norm(zi_2)
            # s2_3=np.linalg.norm(zi_3)
            # s2_4=np.linalg.norm(zi_4)
            # s2_5=np.linalg.norm(zi_5)
            
            zi_1=zi_1/s2_1
            ti=normalize(sketch[6],axis=1)
            
            y_one=np.ones((1,n2))
            
            
            # bi  Pt?
            
            bi1=normalize(np.dot(ti,Pt_1),axis=1)


        
            
            
            Ai_1=np.dot(zi_1,normalize(np.dot(Ps_1,Q_1),axis=0))
            Ai_1=Ai_1/np.linalg.norm(Ai_1).reshape(-1,1)
            # Ai.append(Ai_1)
            # Ai_2=np.dot(zi_2,normalize(np.dot(Ps2,Q2),axis=0))
            # Ai_3=np.dot(zi_3,normalize(np.dot(Ps3,Q3),axis=0))
            # Ai_4=np.dot(zi_4,normalize(np.dot(Ps4,Q4),axis=0))
            # Ai_5=np.dot(zi_5,normalize(np.dot(Ps5,Q5),axis=0))
        
            # Ai_1=Ai_1/np.linalg.norm(Ai_1)
            # Ai_2=Ai_2/np.linalg.norm(Ai_2)
            # Ai_3=Ai_3/np.linalg.norm(Ai_3)
            # Ai_4=Ai_4/np.linalg.norm(Ai_4)
            # Ai_5=Ai_5/np.linalg.norm(Ai_5)
        
            
            omega1=MP(n2,bi1,Ai_1.T,y_one.T)
            # omega.append(omega1)
            # omega2=MP(n2,bi2,Ai_2.T,y_one.T)
            # omega3=MP(n2,bi3,Ai_3.T,y_one.T)
            # omega4=MP(n2,bi4,Ai_4.T,y_one.T)
            # omega5=MP(n2,bi5,Ai_5.T,y_one.T)

    
            
            ai1=np.dot(np.dot(np.dot(Ps_1,Q_1),omega1),Pt_1.T)
            
            # ai2=np.dot(np.dot(np.dot(Ps2,Q2),omega2),Pt2.T)
            # ai3=np.dot(np.dot(np.dot(Ps3,Q3),omega3),Pt3.T)
            # ai4=np.dot(np.dot(np.dot(Ps4,Q4),omega4),Pt4.T)
            # ai5=np.dot(np.dot(np.dot(Ps5,Q5),omega5),Pt5.T)
            
        
            s1_1=np.linalg.norm(np.dot(zi_1,ai1))
            # s1_2=np.linalg.norm(np.dot(zi_2,ai2))
            # s1_3=np.linalg.norm(np.dot(zi_3,ai3))
            # s1_4=np.linalg.norm(np.dot(zi_4,ai4))
            # s1_5=np.linalg.norm(np.dot(zi_5,ai5))
        
            ai1=s1_1/s2_1*ai1
            # ai2=s1_2/s2_2*ai2
            # ai3=s1_3/s2_3*ai3
            # ai4=s1_4/s2_4*ai4
            # ai5=s1_5/s2_5*ai5
        
            # ai.append(ai1)
                
            
            # zi_2=zi_2/s2_2
            # zi_3=zi_3/s2_3
            # zi_4=zi_4/s2_4
            # zi_5=zi_5/s2_5
            zi.append(zi_1.reshape(1, -1))
            
            
            A_all.append(ai1)
            
        zi_all=zi[0].T
        for ii in range (n_dataset_class-1):
            zi_all=np.append(zi_all,zi[ii+1].T,axis=1)
        # zi=np.append(zi,zi_3.T,axis=1)
        # zi=np.append(zi,zi_4.T,axis=1)
        # zi=np.append(zi,zi_5.T,axis=1)
        
        
        # A_all.append(ai3)
        # A_all.append(ai4)
        # A_all.append(ai5)
        
        # n=feature number m=ai colunm number q=number of database num=target number batchnum=q
        # d = 2000   # left feature dimensional number
        N = n_dataset_class   # batch number/right sample number  number  of database
        # k = d  # right feature dimensional number
        num = targetnumber # left sample number  number of class
        
        
        D = utils.normc(A_all[0])
        # Y = utils.normc(ai2)
        d=D.shape[0]
        k = d 
        clf = ODL.ODL(k, lambd=1)
        clf.D = D
        errD = []
        errDX = []
        for ii in range(N-1):
            Y = utils.normc(A_all[ii+1])
            clf.fit(Y, verbose=False, iterations=5)
            
            # errD = np.append(errD, LA.norm(clf.D - D0,'fro')/LA.norm(D0,'fro') )
            # errDX = np.append(errDX, LA.norm(np.dot(clf.D,clf.X ) - D0,'fro' )/LA.norm(D0,'fro') )
        # plt.figure(figsize=(7,5))
        # # plt.plot(np.arange(batchNUM), np.log10(errA), label='ODL')
        # plt.semilogy(np.arange(N), errD, label='D Error')
        # plt.semilogy(np.arange(N), errDX, label='DX Error')
        # plt.legend()
        # plt.show()
        
        
        T = utils.normc(ti.T) # normalize(np.random.normal(0,1, size=(n, num)), axis=0, norm='l2')
        Z = utils.normc(zi_all) # normalize(np.random.normal(0,1, size=(m, q)), axis=0, norm='l2')
        
        # D=np.dot(clf.D,clf.X)
        D=clf.D
        
        B = np.dot(np.dot(np.transpose(T),np.transpose(D ) ),Z )
        Y = np.ones(num).transpose()
        beta = OrthogonalMatchingPursuit(Y, B);

        target=np.dot(np.dot(np.transpose(D ),Z),beta)
        target=target/np.linalg.norm(target)
        target_s[i]=target
        target_i[i]=np.mean(sketch[1],axis=0)/np.linalg.norm(np.mean(sketch[1],axis=0))
        target_GT_i[i]=target_i[i]
        
        T_mean_GT=np.mean(sketch[6],axis=0)
        T_mean_GT=T_mean_GT/np.linalg.norm(T_mean_GT)
        
        sketchfeature6_i=sketchfeature6[sketchlabel6==(list(set(sketchlabel6))[i])]
        T_mean_GT_all=np.mean(sketchfeature6_i,axis=0)
        T_mean_GT_all=T_mean_GT_all/np.linalg.norm(T_mean_GT_all)
        target_GT_s[i]=T_mean_GT_all
    
        
        dist_mean[i]=np.linalg.norm(target-T_mean_GT)
        dist_mean_all[i]=np.linalg.norm(target-T_mean_GT_all)
        print("target number: ",targetnumber," class number:",i ," dist:", dist_mean[i])
        
    class_mean.append(dist_mean_all.copy())
    Average_dist_mean[j]=np.mean(dist_mean)
    Average_dist_mean_all[j]=np.mean(dist_mean_all)
    
    distance_i = cdist(target_i,normalize(sketchfeature1), 'euclidean')


    distance_s = cdist(target_s,normalize(sketchfeature6), 'euclidean')
    retrival_image=np.argsort(distance_i,axis=1)
    retrival_image_2=np.argsort(distance_i,axis=0)[0].reshape(-1,1)
    retrival_sketch=np.argsort(distance_s,axis=0)[0].reshape(-1,1)
    
    sketch_class=np.zeros((retrival_sketch.shape))
    image_class=np.zeros((retrival_image_2.shape))
    for i in range(retrival_sketch.shape[0]):
        sketch_class[i]=class_list[retrival_sketch[i][0]]
    
    for i in range(retrival_image_2.shape[0]):
        image_class[i]=class_list[retrival_image_2[i][0]]
        
    distance2 = np.zeros((str_sim.shape))
    
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



retrival_image=np.argsort(distance_i,axis=1)
retrival_image_2=np.argsort(distance_i,axis=0)[0].reshape(-1,1)
retrival_sketch=np.argsort(distance_s,axis=0)[0].reshape(-1,1)

sketch_class=np.zeros((retrival_sketch.shape))
image_class=np.zeros((retrival_image_2.shape))
for i in range(retrival_sketch.shape[0]):
    sketch_class[i]=class_list[retrival_sketch[i][0]]

for i in range(retrival_image_2.shape[0]):
    image_class[i]=class_list[retrival_image_2[i][0]]
    
accuracy_s=(sketch_class==sketchlabel6.reshape(-1,1)).sum()/sketch_class.shape[0]
accuracy_i=(image_class==sketchlabel1.reshape(-1,1)).sum()/image_class.shape[0]
accuracy_class_s = np.zeros((n_class,1))
accuracy_class_i = np.zeros((n_class,1))
for i in range(n_class):
    s_1=(sketch_class[sketchlabel6==class_list[i]])
    s_2=(sketchlabel6[sketchlabel6==class_list[i]]).reshape(-1,1)
    accuracy_class_s[i]=(s_1==s_2).sum()/s_1.shape[0]
for i in range(n_class):
    s_1=(image_class[sketchlabel1==class_list[i]])
    s_2=(sketchlabel1[sketchlabel1==class_list[i]]).reshape(-1,1)
    accuracy_class_i[i]=(s_1==s_2).sum()/s_1.shape[0]

str_sim = (np.expand_dims(sketchlabel6, axis=1) == np.expand_dims(sketchlabel1, axis=0)) * 1    
distance2 = np.zeros((str_sim.shape))
    
for i in range(distance2.shape[0]):
    distance2[i]=distance_i[retrival_sketch[i]]

# sim1 = 1/(1+distance1)
sim2 = 1/(1+distance2)
nq = str_sim.shape[0]
num_cores = min(multiprocessing.cpu_count(), 32)

# aps1 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim1[iq]) for iq in range(nq))
# map_1 = np.mean(aps1)
# map_1=0.3864

# aps2 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim2[iq]) for iq in range(nq))
# map_2 = np.mean(aps2)
# # map_all[j]=map_2



# plt.figure(figsize=(7,5))
# plt.plot(range(n_target), map_all)
# plt.legend()
# plt.xlabel('training samples per class')
# plt.ylabel('map@all')
# plt.show()



# for k in range(n_class):
#     sketchfeature6_i=sketchfeature6[sketchlabel6==(list(set(sketchlabel6))[k])]
#     T_mean_GT_all=np.mean(sketchfeature6_i,axis=0)
#     T_mean_GT_all=T_mean_GT_all/np.linalg.norm(T_mean_GT_all)
#     target_GT_s[k]=T_mean_GT_all

# result = np.append(normalize(sketchfeature6),target_s,axis=0)
# result = np.append(result,target_GT_s,axis=0)
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

# # class_index_plot=np.argmin(accuracy_class_s)
# class_index_plot=np.argmax(accuracy_class_s)

# # class_index_plot=np.argmin(dist_mean_all_s)
# # class_index_plot=np.argmax(dist_mean_all)



# result=ts_result[0:ts_result.shape[0]-target_s.shape[0]*2][sketchlabel6==class_list[class_index_plot]]
# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(result[:,0], result[:,1], result[:,2], c= 'r', s = 10, marker='.')

# index_mean_ts=ts_result.shape[0]-(target_s.shape[0]-class_index_plot+target_GT_s.shape[0])
# index_mean_GT=ts_result.shape[0]-(target_s.shape[0]-class_index_plot)
# ax.scatter(ts_result[index_mean_ts,0], ts_result[index_mean_ts,1], ts_result[index_mean_ts,2], c= 'b', s = 500, marker='.')
# ax.scatter(ts_result[index_mean_GT,0], ts_result[index_mean_GT,1], ts_result[index_mean_GT,2], c= 'g', s = 500, marker='.')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# red_dot = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
#                           markersize=10, label='feature alignment')
# blue_dot = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
#                           markersize=10, label='estimated mean')
# green_dot = mlines.Line2D([], [], color='green', marker='.', linestyle='None',
#                           markersize=10, label='Ground Truth mean')

# plt.legend(handles=[red_dot, blue_dot, green_dot])
# # plt.show()

# plt.title("t-SNE")
# plt.axis('tight')
# plt.show()


