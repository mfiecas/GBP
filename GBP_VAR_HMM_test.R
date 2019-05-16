library(RcppArmadillo)
library(Rcpp)
library(MASS)
library(abind)
sourceCpp('C:/Users/Brian/Google Drive/PhD Work/GBP_VAR_HMM/GBP_VAR_HMM.cpp')

###### Simulate some data
#Set transition matrices for two simulated subjects
TransMat1 = matrix(c(0.80, 0.1, 0.1, 
                     0.2, 0.80, 0, 
                     0.10, 0.10, 0.80),3,3,byrow=TRUE)
TransMat2 = matrix(c(0.8, 0.2, 0, 
                     0.2, 0.8, 0, 
                     0, 0, 0),3,3,byrow=TRUE)

#Set true emission parameters
A_1=matrix(c(0.8,0,0,0,0.8,0,0,0,0.8),3,3)
Sigma_1=matrix(c(0.35,0,0,0,0.25,0.1,0,0.1,0.35),3,3)

A_2=matrix(c(0.9,0.5,0.4,-0.3,0.6,-0.2,0,0.1,0.8),3,3)
Sigma_2=matrix(c(0.3,0.1,0,0.1,0.2,0.1,0,0.1,0.4),3,3)

A_3=matrix(c(0.7,0,0,0.1,1.1,0.6,-0.1,-0.6,0.3),3,3)
Sigma_3=matrix(c(0.3,0,0,0,0.2,0.1,0,0.1,0.35),3,3)

A_list = list(A_1,A_2,A_3)
Sigma_list = list(Sigma_1,Sigma_2,Sigma_3)

#Set number of epochs and length of epochs for simulated data
Epochs=5
Epoch_length=100

#intitialize time series and state sequences
k1=1
Y1_last = rep(0,3)
k2=1
Y2_last = rep(0,3)
for(i in 1:1000){
  k1 = sample(1:3,1,prob=TransMat1[k1,])
  Y1_last = mvrnorm(1,A_list[[k1]] %*% Y1_last,Sigma_list[[k1]])
  k2 = sample(1:3,1,prob=TransMat2[k2,])
  Y2_last = mvrnorm(1,A_list[[k2]] %*% Y2_last,Sigma_list[[k2]])
}
  
#simulate time series
Y = array(NA,dim=c(3,Epoch_length,Epochs*2))
Z = matrix(NA,Epoch_length,Epochs*2)
for(e in 1:Epochs){
  for(t in 1:Epoch_length){
    k1 = sample(1:3,1,prob=TransMat1[k1,])
    Z[t,e] = k1
    Y1_last = mvrnorm(1,A_list[[k1]] %*% Y1_last,Sigma_list[[k1]])
    Y[,t,e] = Y1_last
    k2 = sample(1:3,1,prob=TransMat2[k2,])
    Z[t,Epochs+e] = k2
    Y2_last = mvrnorm(1,A_list[[k2]] %*% Y2_last,Sigma_list[[k2]])
    Y[,t,Epochs+e] = Y2_last
  }
}
  

############Fit BP-AR-HMM model
AR_order=1
Warmup=50
Burnin=200
Nsamp=500
Fit = BP_AR_HMM(Y, Nsamp, L_min=10, L_max=90, r=AR_order, burnin=Burnin)
Groups=rep(1:2,each=Epochs)
GroupFit = GBP_AR_HMM(Y, Groups, Nsamp, L_min=10, L_max=90, r=AR_order, burnin=Burnin)

  