# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:58:20 2015

@author: Kinara
"""
import numpy as np 
from sklearn import mixture
from scipy.stats import ks_2samp
import math

#create data combining all three parameters 
def create_data(c,ph,orp):
    data = np.zeros((1,3))
    index=range(0,len(c))
    for i in index:
        tmp=[]
        tmp.append(c[i])
        tmp.append(ph[i])
        tmp.append(orp[i])
        tmp = np.array(tmp).reshape(1,-1)
        data = np.vstack((data,tmp))   
    data=data[1:,:]
    return data
    
#create feature vector for a given Window Size,W
#Feature Vector will include (7*3)=21 columns||Number of rows=Length(data)-W.
    #3-Original Data
    #3-Mean of original data 
    #3-Mean of Velocity of data 
    #3-Mean of Acceleration of data 
    #3-Standard Deviation of original data
    #3-Standard Deviation of velocity of data
    #3-Standard Deviation of acceleration of data
def create_feat(c,ph,orp,W):
    feats = []  
    rawdata=create_data(c,ph,orp)
    v=W
    for i in range(W,len(rawdata)):
        dat = rawdata[i-W:i,:]
        vel = np.diff(dat,axis=0)
        acc = np.diff(vel,axis=0)
        ftmp=[]
        datmean = list(np.mean(dat,axis=0))
        velmean = list(np.mean(vel,axis=0))
        accmean = list(np.mean(acc,axis=0))
        datstd = list(np.std(dat,axis=0))
        velstd = list(np.std(vel,axis=0))
        accstd = list(np.std(acc,axis=0))
        ftmp.append(rawdata[v][0])
        ftmp.append(rawdata[v][1])
        ftmp.append(rawdata[v][2])
        v=v+1
        ftmp = ftmp + datmean + velmean + accmean + datstd + velstd + accstd
        feats.append(ftmp)
    feats = np.array(feats)
    return feats,len(rawdata[W:(len(rawdata))]) 

#Assign Value to each record in the dataset. 
#Lowerlimit=low,HigherLimit=high,index=lengthofdataset.      
#i=0 for Conductivity,1 for pH,2 for ORP. 
#Matrix is 2D array[index][i].a is the parameter array, for example : c,ph,orp. 
def assign_value(Matrix,a,low,high,i,index):
    #Value assigned is 0 if val<low, 2 if val>high,1 otherwise. 
    for x in index:
        if(a[x]<low):
            Matrix[x][i]=0
        elif(a[x]>high):
            Matrix[x][i]=2
        else:
            Matrix[x][i]=1
    return Matrix

#Assign State to the given possible combination of cval,pval,oval.
#(1)cval,pval,oval belong to [0,1,2] (2)sval=Value of state to be assigned.
#(3)feats=feature vector (4)label=array storing the states of each point in dataset. 
#(5)dummy=array of dummy values passed by the calling function. 
def assign_states(cval,pval,oval,sval,index,label,Matrix,feats,dummy):
    x=Matrix[:,0]==cval
    #x is a boolean array, has value TRUE where datapoint belongs to given cvalue.
    y=Matrix[:,1]==pval
    #y is a boolean array, has value TRUE where datapoint belongs to given pvalue.
    z=Matrix[:,2]==oval
    #z is a boolean array, has value TRUE where datapoint belongs to given ovalue.
    a=np.ones((1,3),dtype=bool)
    #initialize a.
    for i in index:
        tmp=[]        
        tmp.append(x[i])
        tmp.append(y[i])
        tmp.append(z[i])
        a=np.vstack((a,tmp))  
    ln=len(index)
    #combine 3 boolean arrays x,y,z into a.
    b=np.ones((ln,1),dtype=bool)
    for i in range(1,ln+1):
        b[i-1]=np.all(a[i])
    #b is a boolean array, has value TRUE only if it has value TRUE in all x,y,z arrays.
    ix=[]
    ix=np.where(b==True)[0] 
    #Take indices corresponding to datapoints where b is TRUE.
    if len(ix)==0:
        #If no such points exist, make nstate=0 and initialize data to dummy data.
        nstate=0
        ix2=dummy
    else:
        #Otherwise, assign nstate to length of such points.
        nstate=len(ix)
        #Take data from feature vector that belong to this state. 
        ix2=np.take(feats,ix,axis=0)
        #Assign the state value,sval to all the data points belonging to that state. 
        for i in range(0,len(ix)):
            xx=ix[i]
            label[xx]=sval
        #Return length of datapoints, feature vector, indices corresponding to this state & label. 
    return nstate,ix2,ix,label 
    
#Builds Model corresponding to each state. 
def build_model(Matrix,feats,index):
#Initialize models,label(array storing state),p11(probability distribution),nc(array of states actually existing in the database).
    models=[]
    val=0
    label=np.empty(len(index),dtype=int)
    label.fill(-1)
    p11=[]
    nc=[]
#Initialize dummy(dummy data),ixx(stores feature vector values state wise)
    dummy=range(0,100)
    means=[]
    ixx=[]
    # For 3 parameters and 3 range possibilities(low,high,normal) : 27 states exist.
    for i,j,k in [(i,j,k)for i in range(0,3) for j in range(0,3) for k in range(0,3)]:
        # Run a loop to assign range(0,26) values to each such possibility and get corresponding feat vector.
        nstate,f1,i1,label=assign_states(i,j,k,val,index,label,Matrix,feats,dummy)    
        # Build model only if more than 20 datapoints exist for a given state.
        if nstate>=20:
            # Initialize bic(array storing BIC scores for every model),temp(temporary array storing the built model)
            bic=[]
            temp=[]
            n_components_range = range(1, 7)
            # Find the best value for number of components in the range of 1 to 7.
            for n_components in n_components_range:
                # Run a loop to build model for different values of components.
                m = mixture.GMM(n_components=n_components, covariance_type='full',n_iter=100)
                m.fit(f1)
                temp.append(m)
                # Store the bic values for comparison 
                bic.append(m.bic(f1))
            #Find the model with the minimum bic score.
            ti=bic.index(min(bic))    
            best_gmm=temp[ti] 
            #Use the best model to predict the probability distribution. 
            p11.append(best_gmm.predict_proba(f1)) 
            #Store the state to nc. 
            nc.append(val)
            #Store the best model to the final models array. 
            models.append(best_gmm)
            print i,j,k, "=",val
            #Store the means value to see the means of the different clusters. 
            means.append(best_gmm.means_)
        # IF less than 20 datapoints exist for a given state.
        else:
            #Build model using dummy data. 
            m = mixture.GMM(n_components=1, covariance_type='full',n_iter=100)
            m.fit(dummy)
            models.append(m)
            means.append(m.means_)
            p11.append(m.predict_proba(dummy))
            #Not store the state value in nc. 
        val=val+1
        #Store the data points corresponding to the given state. 
        ixx.append(i1)
    #Function returns the State, probability distribution, models, data points statewise for each state in nc.
    return label,p11,models,ixx,nc

#Function finds substate a given point belongs to using the probability distribution returned by the model.     
def find_substate(p11,ixx,index): 
    #Initialize tt1(array storing substate corresponding to each point)
    tt1=np.empty(len(index),dtype=int)
    tt1.fill(-1)
    number=[]
    #For every possible state:
    for state in range(0,27):
        #If number of datapoints is less than 20, assign default substate value (-3).
        if(len(ixx[state])<20):
            for i in range(0,len(ixx[state])):
                k=ixx[state][i]
                tt1[k]=-3
            #Attach dummy points to the indices array. 
            number.append(range(10))
        #If more than 20 datapoints exist.
        else:
            #Check the probability distribution for the given state to find which component(substate) the model classifies it to. 
            s=p11[state]>0.9
            #Find the number of components built by the model. 
            l=len(p11[state][0]) 
            indices=[]
            #For each such component : 
            for i in range(0,l):
                tt=[]
                ttx=[]
                #Take points where the probability of point belonging to component i is greater than 0.9
                tt=np.where(s[:,i]==True)
                #Take indices coresponding to this component.
                ttx=np.take(ixx[state],tt,axis=0)
                #Store the indices. 
                indices.append(ttx)
                #Assign the sub-state. 
                tt1[ttx]=i
            #Store all the indices state-wise.
            number.append(indices)
        #Function returns array of substates, and datapoints substatewise for each state.
        return tt1,number

#The function builds training model for the given parameters and a window size.
def update_training(c,ph,orp,Window):
    #Call the function to build feature vector. 
    feats,index=create_feat(c,ph,orp,Window)
    index=range(0,index)
    index=np.array(index)
    #Initialize matrix to assign values. 
    Matrix=np.zeros((len(index),3),dtype=int)
    #For conductivity,low=200,high=300.#Can be changed accordingly.
    Matrix=assign_value(Matrix,c,200,300,0,index)
    #For pH,low=7,high=8.#Can be changed accordingly.
    Matrix=assign_value(Matrix,ph,7,8,1,index)
    #For orp,low=100,high=200.#Can be changed accordingly.
    Matrix=assign_value(Matrix,orp,100,200,2,index)
    #Function call to build the model. Pass the feature vector and number of datapoints.
    label,p11,models,ixx,nc=build_model(Matrix,feats,index)
    #Function call to find the state. Pass the parameters returned by build_model.
    tt1,number=find_substate(p11,ixx,index)
    #Function returns information to identify the model built for the given data. 
    return models,label,tt1,p11,index,number,nc
    
# def find_state(Matrix,feats,index,label):
#      dummy=range(0,100)
#      val=0
#      f11=[]
#      ix1=[]
#      nc=[]
#      for i,j,k in [(i,j,k)for i in range(0,3) for j in range(0,3) for k in range(0,3)]:
#         nstate,f1,i1,label=assign_states(i,j,k,val,index,label,Matrix,feats,dummy)
#         if nstate!=0:
#             nc.append(val)            
#         f11.append(f1)
#         ix1.append(i1)
#         val=val+1
#      return label,f11,ix1,nc

def infer_state(Matrix,feats,index,label,n):
    #Initialize values like build_model().
     dummy=range(0,100)
     val=0
     f11=[]
     ix1=[]
     nc=[]
     #Assign states deterministically in a fashion similar to build_model().
     for i,j,k in [(i,j,k)for i in range(0,3) for j in range(0,3) for k in range(0,3)]:
        nstate,f1,i1,label=assign_states(i,j,k,val,index,label,Matrix,feats,dummy)
        if nstate!=0:
            nc.append(val)            
        f11.append(f1)
        ix1.append(i1)
        val=val+1
     for m in range(0,len(label)):
        #Check if new states exist in data that don't in training data: 
         if label[m] not in n:
             x=[]
             for k in n:
                 x.append(abs(k-label[m]))
             #If so,find the closest possible state existing in training data. 
             label[m]=n[x.index(min(x))] 
             #Assign that state. 
     #Returns array of assigned states, feature vector and indices statewise, array of true states in new data.
     return label,f11,ix1,nc
    
def infer_substate(models,p11,f11,n,nc,ix1,index):
    tt2=np.empty(len(index),dtype=int)
    tt2.fill(-1)
    nx=[]
    #If new state doesn't exist in training data, find the closest match for inference.
    for nc1 in nc:
        if nc1 in n:
            nx.append(nc1)
        else:
            x=[]
            for i in n:
                x.append(abs(nc1-i))
            nx.append(n[x.index(min(x))])
    k=0
    #Find feature vector for the required state. Use score_samples method to predict the substate of point.
    for nc1 in nx:
        x=nc[k]
        m=models[nc1]
        ff=f11[x]
        ll=m.score_samples(ff)[1]
        for i in range(0,len(ll)):
            ss=list(ll[i])
            ssv=ss.index(max(ss))
            ssi=ix1[x][i]
            tt2[ssi]=ssv
        k=k+1
    return tt2

#    for nc1 in nc:
#        m=models[nc1]
#        ff=f11[nc1]
#        ll=m.score_samples(ff)[1]
#        for i in range(0,len(ll)):
#            ss=list(ll[i])
#            ssv=ss.index(max(ss))
#            ssi=ix1[nc1][i]
#            tt2[ssi]=ssv
#        return tt2

#Function called to infer the state and substate of the data points arriving realtime. Pass the previous model for inference.
def infer(c,ph,orp,Window,models,p11,n):
    #create_feat,assign_value same as build_model().
    feat1,index=create_feat(c,ph,orp,Window)
    index=range(0,index)
    Matrix=np.zeros((len(index),3),dtype=int)
    Matrix=assign_value(Matrix,c,200,300,0,index)
    Matrix=assign_value(Matrix,ph,7,8,1,index)
    Matrix=assign_value(Matrix,orp,100,200,2,index)
    label1=np.empty(len(index),dtype=int)
    label1.fill(-1)
    #Function call to infer the state. Pass the new feature vector.     
    label1,f11,ix1,nc=infer_state(Matrix,feat1,index,label1,n)
    #Function call to infer the substate. Pass the old p11 and new f11,ix1.
    tt2=infer_substate(models,p11,f11,n,nc,ix1,index)
    #Function returns array of state and substate for every point.  
    return label1,tt2

#Initialize actions array. Use p11 to find the number of substates. 
def initialize_actions(number,p11):
    Actions=[]
    for i in range(0,27):
        zz=np.zeros((len(p11[i][0]),number),dtype=float)
        Actions.append(zz)
        for j in range(0,len(p11[i][0])):
            Actions[i][j]=(1.00/number)
    return Actions

#Find state, substate corresponding to given indice(datapoint)
def get_params(indice,label1,tt2):
    state=label1[indice]
    substate=tt2[indice]
    return state,substate    
    
#Call Function if the action is successful. 
def success(Actions,indice,action,label1,tt2,number):
    state,substate=get_params(indice,label1,tt2)
    Z=Actions[state][substate][action]
    #Calculate weight of remaining actions.
    X=(1-Z)
    #Take its 10%
    Y=(0.1)*X
    #Increase the weight of successful action by 10% weight of remaining actions.
    Actions[state][substate][action]=Z+Y
    g=1-(Y/X)
    #Decrease the weight of remaining actions in proportion to their weights. 
    for i in range(0,number):
        if(i!=action):
            Actions[state][substate][i]=Actions[state][substate][i]*g
    return Actions 

#Call Function if action fails.
def failure(Actions,indice,action,label1,tt2,number):
    state,substate=get_params(indice,label1,tt2)
    X=Actions[state][substate][action]
    Y=(0.1)*X
    #Decrease the weight of failed action by its 10%.
    Actions[state][substate][action]=X*(0.9)
    g=(1+(Y/(1-X)))
    #Increase the weight of remaining actions in proportion to their weights.
    for i in range(0,number):
        if(i!=action):
            Actions[state][substate][i]=Actions[state][substate][i]*g
    return Actions

def update(Actions,indice,action,value,label1,tt2,number):
    if(value):
        Actions=success(Actions,indice,action,label1,tt2,number)
    else:
        Actions=failure(Actions,indice,action,label1,tt2,number)
    return Actions

#Function Called to rebuild Action Matrix after rebuilding model on new data.         
def change(ActionsO,ActionsN,oldn,newn,onum,nnum,oph,nph):
    #Function re-adjusts the substates to accomodate new changes.
    #nx has array of states to which changes were made. #v2 stores indices of identical distributions to swap weights. 
    v2,nx=buildv(oldn,newn,onum,nnum,oph,nph)
    for k in range(0,len(nx)):
        mv=v2[k]
        state=nx[k]
        for m in range(0,len(mv)):
            #Swap action weights according to similarity.
            old=mv[m][0][0]
            new=mv[m][0][1]
            zz=ActionsO[state][old]
            ActionsN[state][new]=zz
    return ActionsN
    
def buildv(oldn,newn,onum,nnum,oph,nph):
    nx=[]
    #Store states existing in both new and old data into nx.
    for i in newn:
        if i in oldn:
            nx.append(i)
    v2=[]
    #For each state in nx:
    for i in nx:
        v1=[]
        n=range(0,len(nnum[i]))
        m=range(0,len(onum[i]))
        for p in m:
            v=[]
            c=[]
            #Use KS test to check if samples come from similar distributions. Null Hypothesis=samples come from similar distributions.
            #Null hypothesis is rejected if D > c(alpha)*root((n1+n2)/(n1*n2)),use alpha=0.05
            for q in n:
                x=onum[i][p][0]
                y=nnum[i][q][0]
                pold=oph[x]
                pnew=nph[y]
                comp=ks_2samp(pold,pnew)[0]
                c.append(comp)
                lo=len(pold)
                ln=len(pnew)
                ccmp=1.36*math.sqrt((lo+ln)/(lo*ln))
                val=[] 
                if(comp<=ccmp):
                    val.append(p) #old_n
                    val.append(q) #new_n
#                    print val
                    v.append(val)
                    v1.append(v)
                    break
            #If no similar distributions exist, find the closest match. 
            if(len(v)==0):
                cc=c.index(min(c))
                val.append(p)
                val.append(cc)
#                print val
                v.append(val)
                v1.append(v)
        v2.append(v1)
    return v2,nx