import os
import random
import copy
import time
import numpy as np
import scipy.io
from classes import nn,Parameter,Performance
from functions import nettestparallel,all, nettrainparallel


data1 = scipy.io.loadmat('sea.mat')
data = data1.get('data')
n_feature = 3


def test_ADL(data, n_feature):   
    (nData, n_column) = data.shape
    M = n_column - n_feature
    preq_data = data[:,0:n_feature]
    preq_label = data[:,n_feature:]  
    chunk_size = 500
    no_of_chunk = int(nData/chunk_size)
    
    drift = {}
    HL = {}
    buffer_x = []
    buffer_T = []
    tTest = []
    tTarget = []
    act = []
    out = []
    #initiate model 
    K = 1 #initial node
    network = nn.nn([n_feature, K, M])
    
    #initiate node evolving iterative parameters
    layer = 1 #initial layer
    parameter = Parameter.Parameter(network, layer,K)
    performance = Performance.Performance()
    
    # initiate drift detection parameter
    alpha_w = 0.0005;
    alpha_d = 0.0001;
    alpha   = 0.0001;
    
    #initiate layer merging iterative parameters
    covariance = np.zeros((1,1,2))
    covariance_old             = covariance
    threshold                  = 0.05
    
    ClassificationRate = {}
    for count in range(0,no_of_chunk):  
        # prepare data
        n = count + 1
        minibatch_data  = preq_data [(n-1)*chunk_size:n*chunk_size]
        minibatch_label = preq_label[(n-1)*chunk_size:n*chunk_size]
        
        # neural network testing      
        print('Chunk: {} of {}'.format(n, no_of_chunk))
        print('Discriminative Testing: running ...')
        parameter.nn.t = n
        [parameter.nn] = nettestparallel.nettestparallel(parameter.nn,minibatch_data,minibatch_label,parameter.ev)
        
        #metrics calculation
        parameter.Loss[n] = parameter.nn.L[parameter.nn.index]
        if(n == 1):
            tTest = parameter.nn.sigma.copy()
            act = parameter.nn.act.copy()
            out = parameter.nn.out.copy()
            parameter.residual_error = np.append(out,parameter.nn.residual_error,axis=0)
        else:
            tTest = np.append(tTest,parameter.nn.sigma,axis=0)
            act = np.append(act,parameter.nn.act,axis=0)
            out = np.append(out,parameter.nn.out,axis=0)
            parameter.residual_error = np.append(out,parameter.nn.residual_error,axis=0)
        parameter.cr[n] = parameter.nn.cr;
        ClassificationRate[n] = np.array(list(parameter.cr.values())).mean()
        print('Classification rate {}'.format(ClassificationRate[n]))
        print('Discriminative Testing: ... finished')
        
        #statistical measure
        performance.ev[n] = {}
        [performance.ev[n]['f_measure'], performance.ev[n]['g_mean'] ,performance.ev[n]['recall'],performance.ev[n]['precision']] = all.performance_summary(parameter.nn.act, parameter.nn.out, M)
        #last chunk only for testing process
        if(n == no_of_chunk):
            print('=========Parallel Autonomous Deep Learning is finished=========')
            break

        #Drift detection: output space
        if(n>1):
            cuttingpoint = 0
            pp = minibatch_label.shape[0]
            F_cut = np.zeros((pp,1))
            F_cut[parameter.nn.bad] = 1
            Fupper = np.max(F_cut)
            Flower = np.min(F_cut)
            miu_F = np.mean(F_cut)
            
            for idx in range(pp):
                cut = idx + 1
                miu_G = np.mean(F_cut[0:cut])
                Gupper = np.max(F_cut[0:cut])
                Glower = np.min(F_cut[0:cut])
                epsilon_G = (Gupper - Glower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha)))
                epsilon_F = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha)))
                if ((epsilon_G + miu_G) >= (miu_F + epsilon_F) and cut<pp):
                    cuttingpoint = cut
                    miu_H = np.mean(F_cut[(cuttingpoint):])
                    epsilon_D = (Fupper - Flower) * np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)) * np.log(1/alpha_d)))
                    epsilon_W = (Fupper - Flower) * np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)) * np.log(1/alpha_w)))
                    break
            if(cuttingpoint == 0):
                miu_H = miu_F
                epsilon_D = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha_d)))
                epsilon_W = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha_w)))
            
            #DRIFT STATUS
            if((np.abs(miu_G - miu_H)) > epsilon_D and cuttingpoint>1):
                st = 1
                print('Drift state: DRIFT')
                layer = layer+1
                parameter.nn.n = parameter.nn.n + 1
                parameter.nn.hl = layer
                print('The new Layer no {} is FORMED around chunk {}'.format(layer, n))
                
                #Initiate NN weight parameters
                ii = parameter.nn.W[layer-1].shape[0]
                parameter.nn.W[layer] = np.random.normal(0,np.sqrt(2/(ii+1)),size = (1, (ii+1)))    
                parameter.nn.vW[layer] = np.zeros((1,ii+1))
                parameter.nn.dW[layer] = np.zeros((1,ii+1))
                
                #Initiate new classifier weight
                parameter.nn.Ws[layer]  = np.random.normal(0,1,size = (M,2))    
                parameter.nn.vWs[layer] = np.zeros((M,2))
                parameter.nn.dWs[layer] = np.zeros((M,2))
                
                #Initiate new voting weight
                parameter.nn.beta[layer] = 1
                parameter.nn.betaOld[layer] = 1
                parameter.nn.p[layer] = 1
                
                # Initiate iterative parameters
                parameter.ev[layer] = {}
                parameter.ev[layer]['layer ']      = layer
                parameter.ev[layer]['kl']          = 0
                parameter.ev[layer]['K']           = 1
                parameter.ev[layer]['cr']           = 0
                parameter.ev[layer]['node']        = {}
                parameter.ev[layer]['miu_NS_old']  = 0
                parameter.ev[layer]['var_NS_old']  = 0
                parameter.ev[layer]['miu_NHS_old'] = 0
                parameter.ev[layer]['var_NHS_old'] = 0
                parameter.ev[layer]['miumin_NS']   = []
                parameter.ev[layer]['miumin_NHS']  = []
                parameter.ev[layer]['stdmin_NS']   = []
                parameter.ev[layer]['stdmin_NHS']  = []
                parameter.ev[layer]['BIAS2']       = {}
                parameter.ev[layer]['VAR']         = {} 
                
                #check buffer
                if(len(buffer_x) == 0):
                    h = parameter.nn.a[len(parameter.nn.a)][:,1:]
                    z = minibatch_label
                else:
                    buffer_x = netffhl(parameter.nn, buffer_x)
                    h = np.append(buffer_x[:,1:],parameter.nn.a[len(parameter.nn.a)][:,1:],axis=0)
                    if(len(buffer_T) == 0):
                        z = np.append(buffer_T,minibatch_label ,axis=0)
                    else:
                        z = minibatch_label
                        
                #Discriminative training for new layer
                print('Discriminative Training for new layer: running ...')
                parameter = nettrainsingle(parameter,h,z)
                print('Discriminative Training for new layer: ... finished')
                buffer_x = []
                buffer_T = []
            elif((np.abs(miu_G - miu_H)) >= epsilon_W and (np.abs(miu_G - miu_H)) < epsilon_D):
                st = 2
                print('Drift state: WARNING')
                buffer_x = minibatch_data
                buffer_T = minibatch_label
            else:
                st = 3
                print('Drift state: STABLE')
                buffer_x = []
                buffer_T = []
        else:
            st = 3
            print('Drift state: STABLE')
            buffer_x = []
            buffer_T = []
        drift[n] = st
        HL[n] = all.checkbeta(parameter.nn.beta)
        parameter.wl[n] = parameter.nn.index
        
        #Discriminative training for winning layer
        if(st != 1):
            print('Discriminative Training: running ...')
            parameter = nettrainparallel.nettrainparallel(parameter, minibatch_label)
            print('Discriminative Training: ... finished')
            
        #Clear current data chunk
        parameter.nn.a = {}
        print('=========Hidden layer number {} was updated========='.format(parameter.nn.index))
    return parameter,performance


if __name__ == "__main__":
    parameter, performance = test_ADL(data,n_feature)
