# -*- coding: utf-8 -*-
"""
Created on Sat May 09 11:32:54 2015

@author: Kinara
"""
import functionsNew1
from functionsNew1 import update_training,initialize_actions,update,infer,get_params,change
from asd import giveVal 

#Call function to get training data.
data = giveVal('','','','2015-01-13')
c13 = data[0]
ph13 = data[1]
orp13 = data[2]

#Call function to train model. Pass training data and window size.
modelsO,stateO,substateO,p11,indexO,numberO,ncO=update_training(c13,ph13,orp13,30)
#Call function to initialize Action Recommender for the given model. 
ActionsO=initialize_actions(5,p11)

#update state substate in db (state0, substate0)

#As soon as real-time data arrives, pass values to infer state and substate for every point.
stateN,substateN=infer(c14,ph14,orp14,30,modelsO,p11,ncO)#real time graph; for each 120 set of values call this.
z=[]
#Get actions corresponding to every point. 
for i in range(0,len(c14)):
    state=stateN[i]
    substate=substateN[i]
    zz=ActionsO[state][substate]
    z.append(zz)
#return stateN substateN zN
    
#One full day data arrived. Get appended data.
data = giveVal(c13,ph13,orp13,'2015-01-14')
c14 = data[0]
ph14 = data[1]
orp14 = data[2]

z=[]
for i in range(0,len(c14)-len(c13)):
    state,substate=get_params(i,stateN,substateN)
    zz=ActionsO[state][substate]
    z.append(zz)

#Pass appended data to rebuild the model. 
modelsN,stateN,substateN,p12,indexN,numberN,ncN=update_training(c14,ph14,orp14,30)
#Initialize ActionsNew for new data.
ActionsN=initialize_actions(5,p12)
#Rebuild ActionsNew according to new and old training data. 
ActionsN=change(ActionsO,ActionsN,ncO,ncN,numberO,numberN,ph13,ph14)

zn=[]
for i in range(0,len(c14)-len(c13)):
    state,substate=get_params(i,stateN,substateN)
    zz=ActionsN[state][substate]
    zn.append(zz)

action_number=4 #Action nunber corresponding to the action user tagged
indice=1000 #point corresponding to end of tag
state,substate=get_params(indice,stateN,substateN)
ActionsN=update(ActionsN,indice,action_number,1,stateN,substateN,5)

