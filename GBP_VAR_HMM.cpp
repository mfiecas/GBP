#include <iostream>
#include <string>
#include "RcppArmadillo.h"
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using namespace std;


const double log2pi = std::log(2.0 * M_PI);
// Definitions of function inputs
//Note that some are defined specifically within certain functions in which case those definitions override these definitions
  //Y is a matrix of observations or a cube of observations with a slice per time series, row per dimension, and column per time point
  //Y_ is a matrix of observations with a row for each dimension and N*T columns (T points from time series 1 followed by T from 2, 3, ...)
  //Y_lags is the design matrix of lagged values (dimension r*DxN*T) corresponding to Y_ and given order
  //D is the dimension of the time series
  //r is the order of the VAR process being considered
  //T is the length of the time series (sometimes set equal to T-r since this is demension of design matrix)
  //K is the current number of states in use (BP) or the truncation level (HDP)
  //K_i is the current number of states in use by time series i
  //K_cur is the number of states currently available in the model
  //K_init is the initial number of states for the (G)BP-AR-HMM
  //K_scale scales the K prior matrix in MNIW prior
  //K_mat = eye()*K_scale in MNIW prior
  //S_0 is the MNIW prior inverse wishart scale matrix
  //n_0 is the MNIW prior inverse wishart degrees of freedom (should be at least D+2)
  //N_samples is the number posterior samples to collect
  //PrintEvery tells the function how many iterations should be run in between printed updates
  //KeepEvery specifies amount of thinning. Sampler runs this many iterations per saved sample
  
  //ProcessPer determines how many loops through HDP/beta-process/HMM parameter sampling are done to improve acceptance (or not sample at all)
  //beta is a vector of global transition parameters for the HDP-HMM model
  //pi is a KxK matrix of state specific transition distributions with a row for each state
  //kappa is the sticky parameter for the sticky HMM transition prior
    //a_kappa and b_kappa are the gamma hyperparameters for kappa
    //sigma_kappa_sq is the variance of the gamma proposal distribution for kappa
  //gamma is the concentration parameter for the sticky HMM transition prior
    //a_gamma and b_gamma are the gamma hyperparameters for gamma
    //sigma_gamma_sq is the variance of the gamma proposal distribution for gamma
  //lambda is a matrix with gamma+kappa on the diagonals and gamma elsewhere
  //alpha is either the HDP-HMM beta process parameter
    //or it is the BP-VAR upper-level parameter
    //a_alpha and b_alpha are the gamma hyperparameters for alpha
  //c is a beta process hyperparameter (set to 1 for typical IBP)
    //a_c and b_c are the gamma hyperparameters for c
    //sigma_c_sq is the variance of the gamma proposal distribution for c
    
  //log_Likeli is a K_curxT matrix of likelihoods for time series i at time t for state k, sometimes it is a cube with a slice per time series
  //eta_i is a matrix where the k-th row is proportional to the transition probabilities from state k
  //eta can also be a cube where slice i is eta_i for time series (or group) i
  //Transmat is an KxK matrix where the (i,j) entry contains the number of transitions from state i to state j
  //Transcube is a cube of TransMats where the i-th slice is the transmat from time series (or group) i
  //target_F is a matrix of feature allocations assignments to get probabilities for in split step if specified (otherwise F is sampled)
  //F is a matrix where the ith column indicates which features times series i possesses (F.col(i) = f_i)
  //f_i is a vector indicating which features time series i possesses
  //f_i_prop is a new proposal for f_i
  //m is a vector where the k-th element is how many time series (or groups) possess feature k
  //m_i = m - f_i
  //Z is a TxN matrix with the state sequences for subject i in column i
  //vecZ = vectorise(Z) contains all of the state sequences of all the time series concatenated
  //target_Z is a vector of target state assignments to get probabilities for if specified (otherwise Z is sampled)
  //n_k is the number of time points in state k (if integer) or the k-th element is the number of time points in state k (if vector)
  //n_k_i is a matrix where the (k,i) element is the number of time points from time series (group) i in state k
  //Marg_Prior and Lik_base are cached values used in the likelihood calculations
  
  //S_gb, S_bb_inv, and S_b are the sufficient statistics
  //logdet_S_gb(_prop) and logdet_S_bb(_prop) are the log determinants of the corresponding sufficient statistics
  
  //BDper determines how many loops through the birth/death process per time series (or group) are done per iteration
  //SMper determines how many split/merge moves are proposed per iteration
  //L_min is the minimum window length considered for birth/death steps
  //L_max is the maximum window length considered for birth/death steps
  //W_0 is the start of the chosen window of data for birth/death steps
  //W_1 is the end of the chosen window of data for birth/death steps
  //anneal_temp is the annealing temperature to improve mixing during burn-in for birth/death and split/merge steps. =1 after burn-in
  //i and j refer to time series i an j
  //k_i and k_j refer to potential selected features k_i and k_j  for split/merge (if not specified then they are sampled by FeatureSelect)
  
  //group_i is a vector of indices of the time series in group i
  //N_groups is the total number of groups being considered for GDP-AR-HMM
  //N_g is the number of time series in a specific group (simetimes an integer sometimes a vector of length N_groups)
  

//function to calculate factorial on an integer
// [[Rcpp::export]]
int factorial(int x, int result = 1) {
  if (x == 1) return result; else return factorial(x - 1, x * result);
}


//function to calculate log multivariate gamma (dimension=D) function of x
// [[Rcpp::export]]
double logMultiGamma(double x, int D) {
  double out = D*(D-1)/4*log(M_PI);
  for(int d=1; d<=D; d++){
    out += lgamma(x+(1-d)/2);
  }
  return out;
}


//function to calculate multivariate normal (log-)density of x given mean and Sigma
// [[Rcpp::export]]
double dmvnrm_arma(const rowvec& x, const rowvec& mean, const mat& sigma, bool log_lik=true){
  int xdim = x.n_cols;
  mat rooti = trans(inv(trimatu(chol(sigma))));
  double rootisum = sum(log(rooti.diag()));
  double constants = -(static_cast<double>(xdim)/2.0) * log(2.0 * M_PI);
  vec z = rooti*trans(x - mean);
  if(log_lik == true){
    return(constants - 0.5 * sum(z%z) + rootisum);
  }else{
    return(exp(constants - 0.5 * sum(z%z) + rootisum));
  }
}



//function to calculate multivariate normal (log-)density of x given mean and Sigma
// [[Rcpp::export]]
mat getLikeliMat(const mat& Y_, const mat& Y_lags, const cube& A, const cube& Sigma, 
                 int n, const vec& f_i, int K_cur, int T, int D){
  //A and Sigma have K_i = sum(f_i) slices
  mat log_Likeli(K_cur, T);
  log_Likeli.fill(-pow(10,323));
  mat Y_hat(D, T);
  int counter = 0;
  for(int k=0; k<K_cur; k++){
    if(f_i(k) == 1){
      Y_hat = A.slice(counter)*Y_lags.cols(n*T,n*T+T-1);
      for(int t=0; t<T; t++){
        log_Likeli(k, t) = dmvnrm_arma(Y_.col(n*T+t).t(), Y_hat.col(t).t(), Sigma.slice(counter));
      }
      counter++;
    }
  }
  
  return log_Likeli;
}


//Function sample from VAR(r) model with MNIW conjugate prior
// [[Rcpp::export]]
List VAR_MNIW(const mat& Y, const int& r, const int& N_samples, double K_scale=1){
  //Y is a matrix containing the multivariate time series with a row for each dimension
  
  int D = Y.n_rows;
  double T = Y.n_cols;
  
  cube A(D, D*r, N_samples);
  cube Sigma(D, D, N_samples);
  mat K_mat = K_scale*eye(D*r, D*r);
  
  mat Y_lags(D*r,T-r);
  for(int ord=0; ord<r; ord++){
    for(int t=r; t<T; t++){
      Y_lags.submat(ord*D, t-r, (ord+1)*D-1, t-r) = Y.col(t-(ord+1));
    }
  }
  
  mat S_bb_inv = inv(Y_lags * Y_lags.t() + K_mat);
  mat S_b =  Y.tail_cols(T-r) * Y_lags.t();
  mat S = Y.tail_cols(T-r) * Y.tail_cols(T-r).t();
  mat S_gb = S - S_b * S_bb_inv * S_b.t();
  vec Y_bar = mean(Y,1);
  mat Sigma_Bar(D,D,fill::zeros);
  for(int t=0; t<T; t++){
    Sigma_Bar += (Y.col(t)-Y_bar)*(Y.col(t)-Y_bar).t();
  }
  
  mat S_0 = 0.75/T*Sigma_Bar;
  int n_0 = D*r + 2;
  mat IWish_Dinv = chol(inv(S_gb + S_0));
  
  vec PostMean = vectorise(S_b*S_bb_inv);
  
  for(int iter=0; iter<N_samples; iter++){
    Sigma.slice(iter) = iwishrnd(S_gb + S_0, n_0 + T, IWish_Dinv);
    A.slice(iter) = reshape(mvnrnd(PostMean, kron(S_bb_inv, Sigma.slice(iter))), D, D*r);
  }
  
  return List::create(
    _["Sigma"] = Sigma,
    _["A"] = A
  );
}


//Function to sample a single iteration from the HDP
// [[Rcpp::export]]
List Sample_HDP(int K, const mat& TransMat, vec& beta, const mat& pi, 
                double alpha, double kappa, double gamma, double a_hyper,
                double b_hyper, double c_hyper, double d_hyper, int ProcessPer){

  //initialize some stuff
  vec beta_new(K);
  mat pi_new(K,K);
  vec n(K,fill::zeros);
  vec m(K,fill::zeros);
  vec w(K);
  rowvec dir_temp(K);
  double p_temp;
  double rho = kappa/(alpha+kappa);
  
  //loop through all transitions to calculate m(j), # of tables assigned dish j
  for(int i=0; i<K; i++){
    for(int j=0; j<K; j++){
      for(int k=0;k<TransMat(i,j);k++){
        if(i==j){
          p_temp = (alpha*beta(j) + kappa) / (n(j) + alpha*beta(j) + kappa);
        }else{
          p_temp = alpha*beta(j) / (n(j) + alpha*beta(j));
        }
        m(j) += R::rbinom(1,p_temp);
        n(j)++;
      }
    }
  }
  
  //calculate m dot dot for posteriors
  double m_dd=sum(m);
  
  //calculate override variables w and substract them from m to get m bar
  for(int i=0; i<K; i++){
    w(i) = R::rbinom(m(i), rho/(rho + beta(i)*(1-rho)));
    m(i) = m(i) - w(i);
  }
  
  //calculate the sum of w and the sum of m bar for posteriors
  double w_sum = sum(w); 
  double m_bdd = sum(m);
  
  
  //draw new betas from Dir posterior using gamma draws
  for(int i=0; i<K; i++){
    dir_temp(i) = R::rgamma(m(i)+gamma/K,1);
  }
  beta_new = (dir_temp/sum(dir_temp)).t();
  
  //draw new pis from Dir posterior using gamma draws and new betas
  for(int i=0; i<K; i++){
    for(int j=0; j<K; j++){
      if(i==j){
        dir_temp(j) = R::rgamma(alpha*beta_new(j)+TransMat(i,j)+kappa,1) + pow(10,-323);
      }else{
        dir_temp(j) = R::rgamma(alpha*beta_new(j)+TransMat(i,j),1) + pow(10,-323);
      }
    }
    pi_new.row(i) = dir_temp/sum(dir_temp);
  }
  
  //initialize auxiliary variables r and s used to sample hyperparameters
  vec r_temp(K);
  vec s_temp(K);
  double alpha_plus_kappa_new;
  double eta_temp;
  double zeta_temp;
  double gamma_new = gamma;
  double rho_new;
  double kappa_new = kappa;
  double alpha_new = alpha;
  for(int ProcIter=0; ProcIter<ProcessPer; ProcIter++){
    for(int i=0; i<K; i++){
      r_temp(i) = R::rbeta(alpha+kappa+1,n(i));
      s_temp(i) = R::rbinom(1,n(i)/(n(i)+alpha+kappa));
    }
    //draw new (alpha+kappa) from posterior
    alpha_plus_kappa_new = R::rgamma(a_hyper+m_dd-sum(s_temp), 
                                     b_hyper-sum(log(r_temp)));
    //sample auxiliary variables eta and zeta used in gamma posterior
    eta_temp = R::rbeta(gamma+1,m_bdd);
    zeta_temp = R::rbinom(1,m_bdd/(m_bdd+gamma));
    //sample new gamma value from posterior
    gamma_new = R::rgamma(a_hyper+K-zeta_temp, b_hyper-log(eta_temp));
    
    //sample new rho from posterior and back out new kappa and alpha values
    rho_new = R::rbeta(w_sum+c_hyper, m_dd-w_sum+d_hyper);
    kappa_new = rho_new*alpha_plus_kappa_new;
    alpha_new = alpha_plus_kappa_new - kappa_new;
  }
  
  //return updated parameters
  return List::create(
    _["beta_new"] = beta_new,
    _["pi_new"] = pi_new,
    _["alpha_new"] = alpha_new,
    _["gamma_new"] = gamma_new,
    _["kappa_new"] = kappa_new
  );
}


//Function to converted log_p vec where p is proportion to multinom probs to a multinom prob vector of length K
// [[Rcpp::export]]
double log_sum(vec log_p, int K, vec f){
  double MaxVal = max(log_p);
  double temp = 0;
  for(int k=0; k<K; k++){
    if(f(k) == 1){temp += exp(log_p(k) - MaxVal);}
  }
  return (MaxVal + log(temp));
}


//Function to converted log_p vec where p is proportion to multinom probs to a multinom prob vector of length K
// [[Rcpp::export]]
vec Multinom_prob_stable(vec log_p, int K, vec f){
  vec probs(K);
  if(max(log_p) < -pow(10,300)){
    for(int k=0; k<K; k++){if(f(k) == 0){probs(k)=0;}else{probs(k)=1;}}
  }else{
    double denom = log_sum(log_p, K, f);
    for(int k=0; k<K; k++){
      if(f(k) == 0){probs(k) = 0;}else{
        probs(k) = exp(log_p(k) - denom) + pow(10,-323);
      }
    }
  }
  
  return probs/sum(probs);
}


//Function to sample from a multinomial dist with probs p and K many categories
// [[Rcpp::export]]
int Sample_Multinom(vec p){
  vec cumProbs = cumsum(p);
  double temprand = randu();
  int check = 0;
  int counter = 0;
  int val;
  while(check == 0){
    if(temprand < cumProbs(counter)){
      val = counter;
      check = 1;
    }
    counter++;
  }
  
  //return updated parameters
  return val;
}


//Function to sample a state sequence from the HMM
// [[Rcpp::export]]
List Sample_Z(const mat& log_Likeli, const mat& eta_i, int K_cur, 
              const vec& f_i, int T){
  
  uvec avail = find(f_i == 1);
  int K_i = avail.n_elem;
  vec n_k_new(K_cur,fill::zeros);
  mat log_pi_i(K_i,K_i);
  mat eta_temp = eta_i.submat(avail, avail);
  for(int k=0; k<K_i; k++){
    log_pi_i.row(k) = log(eta_temp.row(k) / sum(eta_temp.row(k)));
  }
  
  //calculate backward messages using likelihoods
  mat Mes(K_i,T);
  Mes.col(T-1).fill(0);
  vec Ones(K_i,fill::ones);
  vec Maxes(K_i);
  uvec t_vec(1);
  for(int t=T-2; t>=0; t--){
    t_vec(0) = t+1;
    for(int k=0; k<K_i; k++){
      Mes(k,t) = log_sum(log_pi_i.row(k).t() + log_Likeli.submat(avail, t_vec) + 
                         Mes.col(t+1), K_i, Ones);
    }
    Maxes.fill(max(Mes.col(t)));
    Mes.col(t) = Mes.col(t) - Maxes;
  }
  
  //sample new state assignments
  vec Z_new(T);
  vec p(K_i);
  mat TransMat(K_cur,K_cur,fill::zeros);
  int Z_temp = -1;
  double log_prob=0;
  for(int t=0; t<T; t++){
    t_vec(0) = t;
    if(t > 0){
      p = Multinom_prob_stable(log_Likeli.submat(avail, t_vec) + Mes.col(t) + 
                               log_pi_i.row(Z_temp).t(), K_i, Ones);
    }else{
      p = Multinom_prob_stable(log_Likeli.submat(avail, t_vec) + Mes.col(t), K_i, Ones);
    }
    Z_temp = Sample_Multinom(p);
    Z_new(t) = avail(Z_temp);
    if(t > 0){TransMat(Z_new(t-1),Z_new(t))++;}
    n_k_new(Z_new(t))++;
    log_prob += log(p(Z_temp));
  }
  
  //return updated parameters
  return List::create(
    _["Z_new"] = Z_new,
    _["TransMat"] = TransMat,
    _["n_k_new"] = n_k_new,
    _["log_prob"] = log_prob
  );
}


//Function to get log probability of sampling a given state sequence from the HMM
// [[Rcpp::export]]
double Sample_Z_prob(const mat& log_Likeli, const mat& eta_i, int K_cur, 
                     const vec& f_i, int T, const vec& target_Z){
  
  uvec avail = find(f_i == 1);
  int K_i = avail.n_elem;
  vec key(K_cur);
  key.fill(-1);
  for(int k=0; k<K_cur; k++){
    uvec temp_ind = find(avail == k);
    if(temp_ind.n_elem>0){key(k) = temp_ind(0);}
  }
  
  mat log_pi_i(K_i,K_i);
  mat eta_temp = eta_i.submat(avail, avail);
  for(int k=0; k<K_i; k++){
    log_pi_i.row(k) = log(eta_temp.row(k) / sum(eta_temp.row(k)));
  }
  
  //calculate backward messages using likelihoods
  mat Mes(K_i,T);
  Mes.col(T-1).fill(0);
  vec Ones(K_i,fill::ones);
  vec Maxes(K_i);
  uvec t_vec(1);
  for(int t=T-2; t>=0; t--){
    t_vec(0) = t+1;
    for(int k=0; k<K_i; k++){
      Mes(k,t) = log_sum(log_pi_i.row(k).t() + log_Likeli.submat(avail, t_vec) + 
        Mes.col(t+1), K_i, Ones);
    }
    Maxes.fill(max(Mes.col(t)));
    Mes.col(t) = Mes.col(t) - Maxes;
  }
  
  //sample new state assignments
  vec p(K_i);
  int Z_temp = -1;
  double log_prob=0;
  for(int t=0; t<T; t++){
    t_vec(0) = t;
    if(t > 0){
      p = Multinom_prob_stable(log_Likeli.submat(avail, t_vec) + Mes.col(t) + 
                               log_pi_i.row(Z_temp).t(), K_i, Ones);
    }else{
      p = Multinom_prob_stable(log_Likeli.submat(avail, t_vec) + Mes.col(t), K_i, Ones);
    }
    Z_temp = key(target_Z(t));
    log_prob += log(p(Z_temp));
  }
  
  return log_prob;
}


//Function to calculate MNIW sufficient statistics given data in state k
// [[Rcpp::export]]
List get_MNIW_SS(const mat& Y, const mat& Y_lags, const mat& K_mat, int D, int r){
  //Y is a matrix containing the multivariate time series with a row for each dimension
  //Y_lags is the design matrix corresponding to Y
  
  mat S_bb(D*r,D*r);
  mat S_bb_inv(D*r,D*r);
  mat S_b(D,D*r);
  mat S_gb(D,D);
  if(Y.n_cols > 0){
    S_bb = Y_lags * Y_lags.t() + K_mat;
    S_bb_inv = inv(S_bb);
    S_b =  Y * Y_lags.t();
    S_gb = Y * Y.t() - S_b * S_bb_inv * S_b.t();
  }else{
    S_bb = K_mat;
    S_bb_inv = inv(S_bb);
    S_b.fill(0);
    S_gb.fill(0);
  }
  
  return List::create(
    _["S_bb_inv"] = S_bb_inv,
    _["S_bb"] = S_bb,
    _["S_b"] = S_b,
    _["S_gb"] = S_gb
  );
}


//Function to sample from VAR(r) model with MNIW conjugate prior
// [[Rcpp::export]]
List Sample_MNIW(int D, int r, const mat& S_gb, const mat& S_bb_inv, 
                 const mat& S_b, const mat& S_0, int n_0, int n_k){
  //n_k is the total number of time points assigned to state k
  
  mat A_new(D, D*r);
  mat Sigma_new(D, D);
  mat IWish_Dinv = chol(inv(S_gb + S_0));
  vec PostMean = vectorise(S_b*S_bb_inv);
  
  Sigma_new = iwishrnd(S_gb + S_0, n_0 + n_k, IWish_Dinv);
  A_new = reshape(mvnrnd(PostMean, kron(S_bb_inv, Sigma_new)), D, D*r);
  
  return List::create(
    _["Sigma_new"] = Sigma_new,
    _["A_new"] = A_new
  );
}




//Function to fit Sticky HDP VAR HMM model with MNIW conjugate prior
// [[Rcpp::export]]
List Sticky_HDP_HMM(mat Y, int N_samples, int r=1, int K=20, int ProcessPer=1,
                    double alpha=1, double kappa=100, double gamma=1,
                    double a_hyper=1, double b_hyper=1, double c_hyper=10, 
                    double d_hyper=1, double K_scale=1, int n_0=0, 
                    int PrintEvery=100){
  //a_hyper and b_hyper are the gamma hyper parameters for (alpha+kappa) and gamma
    //Fox used a=1 and b=0.01
  //c_hyper and d_hyper are the beta hyper parameters for rho where rho=kappa/(alpha+kappa)
    //Fox used c=10 and d=1

  int D = Y.n_rows;
  double T = Y.n_cols;
  //construct design matrix from lagged observations of dimension D*r x T-r
  mat Y_lags(D*r,T-r);
  for(int t=0; t<(T-r); t++){
    Y_lags.col(t) = vectorise(fliplr(Y.cols(t, t+r-1)));
  }
  
  //remove first r observations since we do not have lagged observations for them
  T = T-r;
  Y.shed_cols(0,r-1);
  
  //set priors to match those of Fox
  mat Sigma_Bar(D,D,fill::zeros);
  vec Y_bar = mean(Y.cols(1,T-1)-Y.cols(0,T-2),1);
  for(int t=1; t<T; t++){
    Sigma_Bar += (Y.col(t)-Y.col(t-1)-Y_bar)*(Y.col(t)-Y.col(t-1)-Y_bar).t();
  }
  mat S_0 = 0.75/T*Sigma_Bar;
  if(n_0 == 0){
    n_0 = D*r + 2;
  }
  
  mat K_mat = K_scale*eye(D*r, D*r);
  
  //initialize emission parameters
  field<cube> A(N_samples);
  cube A_temp(D, D*r, K);
  field<cube> Sigma(N_samples);
  cube Sigma_temp(D, D, K);
  
  List outS = get_MNIW_SS(Y, Y_lags, K_mat, D, r);
  mat S_gb = as<mat>(outS("S_gb"));
  mat S_bb_inv = as<mat>(outS("S_bb_inv"));
  mat S_b = as<mat>(outS("S_b"));
  for(int k=0; k<K; k++){      
    List out_temp = Sample_MNIW(D, r, S_gb, S_bb_inv, S_b, S_0, n_0, floor(T/K));
    A_temp.slice(k) = as<mat>(out_temp("A_new"));
    Sigma_temp.slice(k) = as<mat>(out_temp("Sigma_new"));
  }
  A(0) = A_temp;
  Sigma(0) = Sigma_temp;
  
  //calculate likelihoods for each observation for each state
  vec f_temp(K,fill::ones);
  mat log_Likeli = getLikeliMat(Y, Y_lags, A(0), Sigma(0), 0, f_temp, K, T, D);
  
  //initialize global transition parameters beta
  vec beta(K);
  vec dir_temp(K);
  for(int j=0; j<K; j++){
    dir_temp(j) = R::rgamma(gamma/K,1);
  }
  beta = dir_temp/sum(dir_temp);
  
  //initialize state specific transition parameters pi
  mat pi(K,K);
  for(int i=0; i<K; i++){
    for(int j=0; j<K; j++){
      dir_temp(j) = R::rgamma(alpha*beta(j),1) + pow(10,-323);
    }
    pi.row(i) = (dir_temp/sum(dir_temp)).t();
  }

  //initialize state sequences, Z
  mat Z(N_samples, T);
  mat TransMat(K, K);
  vec n_k(K);
  
  vec f_i(K,fill::ones);
  List outZ = Sample_Z(log_Likeli, pi, K, f_i, T);
  Z.row(0) = as<rowvec>(outZ("Z_new"));
  
  field<uvec> inK(K);

  for(int n_samples=1; n_samples<N_samples; n_samples++){
    
    outZ = Sample_Z(log_Likeli, pi, K, f_i, T);
    Z.row(n_samples) = as<rowvec>(outZ("Z_new"));
    TransMat = as<mat>(outZ("TransMat"));
    n_k = as<vec>(outZ("n_k_new"));
    for(int k=0; k<K; k++){
      uvec inK2 = find(Z.row(n_samples).t() == k);
      inK(k) = inK2;
    }
    
    List out5 = Sample_HDP(K, TransMat, beta, pi, alpha, kappa, gamma, 
                           a_hyper, b_hyper, c_hyper, d_hyper, ProcessPer);
    beta = as<vec>(out5("beta_new"));
    pi = as<mat>(out5("pi_new"));
    alpha = as<double>(out5("alpha_new"));
    kappa = as<double>(out5("kappa_new"));
    gamma = as<double>(out5("gamma_new"));
    
    for(int k=0; k<K; k++){
      outS = get_MNIW_SS(Y.cols(inK(k)), Y_lags.cols(inK(k)), K_mat, D, r);
      S_gb = as<mat>(outS("S_gb"));
      S_bb_inv = as<mat>(outS("S_bb_inv"));
      S_b = as<mat>(outS("S_b"));
      List out_temp = Sample_MNIW(D, r, S_gb, S_bb_inv, S_b, S_0, n_0, n_k(k));
      A_temp.slice(k) = as<mat>(out_temp("A_new"));
      Sigma_temp.slice(k) = as<mat>(out_temp("Sigma_new"));
    }
    log_Likeli = getLikeliMat(Y, Y_lags, A_temp, Sigma_temp, 0, f_temp, K, T, D);
    A(n_samples) = A_temp;
    Sigma(n_samples) = Sigma_temp;
    
    if((n_samples+1)%PrintEvery == 0){cout << n_samples+1 << " posterior samples complete" << endl;}
  }
  
  return List::create(
    _["Sigma"] = Sigma,
    _["A"] = A,
    _["Z"] = Z
  );
}




//Function to get likelihood of Y_i given f_i and theta with state sequences integrated out 
// [[Rcpp::export]]
double get_L_Y_given_f_theta(vec f_i, int K_cur, int T, mat eta, mat log_Likeli){
  
  mat log_pi(K_cur,K_cur);
  vec Mes(K_cur);
  double L;
  
  for(int k=0; k<K_cur; k++){
    log_pi.row(k) = log(eta.row(k) % f_i.t() / sum(eta.row(k) % f_i.t()));
  }
  
  Mes = log_Likeli.col(0) % f_i;
  double s_temp = log_sum(Mes, K_cur, f_i);
  vec s_temp_vec(K_cur);
  s_temp_vec.fill(s_temp);
  vec MesLast = Mes - (s_temp_vec % f_i);
  L = s_temp;
    
  for(int t=1; t<T; t++){
    for(int k=0; k<K_cur; k++){
      if(f_i(k) == 1){
        Mes(k) = log_Likeli(k,t) + log_sum(log_pi.row(k).t() + MesLast, K_cur, f_i);
      }
    }
    s_temp = log_sum(Mes, K_cur, f_i);
    s_temp_vec.fill(s_temp);
    MesLast = Mes - (s_temp_vec % f_i);
    L += s_temp;
  }
  
  return L;
}


//Function to sample feature indicators
// [[Rcpp::export]]
vec sample_F(vec f_i, int K_cur, int N, int T, int D, const vec& m_i, 
             double c, const mat& eta_i, const mat& log_Likeli_i){
  
  vec f_star(K_cur);
  double L = get_L_Y_given_f_theta(f_i, K_cur, T, eta_i, log_Likeli_i);
  vec Mes(K_cur);
  
  for(int k=0; k<K_cur; k++){
    f_star = f_i;
    if(m_i(k)>0){
      if(f_i(k)==sum(f_i)){f_i(k)=1;}else{
        if(f_i(k) == 0){
          f_star(k) = 1;
        }else{
          f_star(k) = 0;
        }
        double P_on =  m_i(k) / (N + m_i(k) + c);
        double P_star = log(f_star(k)*P_on + (1-f_star(k))*(1-P_on));
        double P =log(f_i(k)*P_on + (1-f_i(k))*(1-P_on));
        double L_star = get_L_Y_given_f_theta(f_star, K_cur, T, eta_i, log_Likeli_i);
        if(log(randu()) < (P_star+L_star-P-L)){
          f_i = f_star;
          L = L_star;
        }
      }
    }
  }
  
  return f_i;
}


//Function to sample transition probabilities
// [[Rcpp::export]]
mat sample_eta(const mat& TransMat_i, double gamma, double kappa, int K_cur, int K_i){
  
  vec temp(K_cur);
  mat eta(K_cur,K_cur);
  rowvec Mins(K_cur);
  Mins.fill(pow(10,-323));
  double scale;
  
  for(int k=0; k<K_cur; k++){
    for(int j=0; j<K_cur; j++){
      if(k==j){
        temp(j) = R::rgamma(gamma+TransMat_i(k,j)+kappa,1) + pow(10,-323);
      }else{
        temp(j) = R::rgamma(gamma+TransMat_i(k,j),1) + pow(10,-323);
      }
    }
    scale = R::rgamma(gamma*K_i+kappa,1);
    eta.row(k) = scale*(temp/sum(temp)).t() + Mins;
  }
  
  return eta;
}


//function to sample alpha hyperparameter for the beta process
double sample_alpha(double a_alpha, double b_alpha, double c, int K_cur, int N){
  
  double alpha_new;
  double a_new = a_alpha + K_cur;
  double b_new = b_alpha;
  for(int n=0; n<N; n++){
    b_new += c/(c+n);
  }
  
  alpha_new = R::rgamma(a_new, b_new) + pow(10,-323);
  
  return alpha_new;
}


//function to sample c hyperparameter for the beta process
double sample_c(double c, double a_c, double b_c, double alpha, double sigma_c_sq, 
              int K_cur, int N, vec m){
  
  double a = pow(c,2) / sigma_c_sq;
  double b = a / c;
  
  double c_prop = R::rgamma(a, b) + pow(10,-323);
  double a_prop = pow(c_prop,2) / sigma_c_sq;
  
  double c_term = 0;
  double c_term_prop = 0;
  for(int n=0; n<N; n++){
    c_term += c/(c+n);
    c_term_prop += c_prop/(c_prop+n);
  }
  
  double l = K_cur*log(c) - alpha*c_term;
  double l_prop = K_cur*log(c_prop) - alpha*c_term_prop;

  for(int k=0; k<K_cur; k++){
    l += lgamma(N-m(k)+c) - lgamma(N+c);
    l_prop += lgamma(N-m(k)+c_prop) - lgamma(N+c_prop);
  }
  
  double MH_ratio = l_prop - l + (a_c+a_prop-a)*log(c_prop) - 
                    (a_c+a-a_prop)*log(c) - (c_prop-c)*b_c +
                    lgamma(a) - lgamma(a_prop) + (a-a_prop)*log(sigma_c_sq);
  
  if(log(randu()) < MH_ratio){
    c = c_prop;
  }
  
  return c;
}


//function to sample gamma hyperparameter for the HMM transitions
double sample_gamma(double gamma, double a_gamma, double b_gamma, 
                    double sigma_gamma_sq, int N, int K_cur, double kappa, 
                    const mat& F, const cube& eta){
 
  double a = pow(gamma,2) / sigma_gamma_sq;
  double b = a / gamma;
  
  double gamma_prop = R::rgamma(a, b) + pow(10,-323);
  double a_prop = pow(gamma_prop,2) / sigma_gamma_sq;
  
  double l = 0;
  double l_prop = 0;
  double K_n;
  mat pi_n(K_cur,K_cur);
  for(int n=0; n<N; n++){
    K_n = sum(F.col(n));
    for(int k=0; k<K_cur; k++){
      pi_n.row(k) = eta.slice(n).row(k) % F.col(n).t() / sum(eta.slice(n).row(k) % F.col(n).t());
    }
    l += lgamma(gamma*K_n+kappa) - (K_n-1)*lgamma(gamma) - lgamma(gamma+kappa);
    l_prop += lgamma(gamma_prop*K_n+kappa) - (K_n-1)*lgamma(gamma_prop) - 
              lgamma(gamma_prop+kappa);
    for(int k=0; k<K_cur; k++){
      for(int k2=0; k2<K_cur; k2++){
        if(F(k,n)==1 && F(k2,n)==1){
          if(k==k2){
            l += (gamma+kappa-1)*log(pi_n(k,k2));
            l_prop += (gamma_prop+kappa-1)*log(pi_n(k,k2));
          }else{
            l += (gamma-1)*log(pi_n(k,k2));
            l_prop += (gamma_prop-1)*log(pi_n(k,k2));
          }
        }
      }
    }
  }
  
  double MH_ratio = l_prop - l + (a_gamma+a_prop-a)*log(gamma_prop) - 
    (a_gamma+a-a_prop)*log(gamma) - (gamma_prop-gamma)*b_gamma +
    lgamma(a) - lgamma(a_prop) + (a-a_prop)*log(sigma_gamma_sq);
  
  if(log(randu()) < MH_ratio){
    gamma = gamma_prop;
  }
  
  return gamma;
}


//function to sample kappa hyperparameter for the HMM transitions
double sample_kappa(double kappa, double a_kappa, double b_kappa, 
                    double sigma_kappa_sq, int N, int K_cur, double gamma, 
                    const mat& F, const cube& eta){
  
  double a = pow(kappa,2) / sigma_kappa_sq;
  double b = a / kappa;
  
  double kappa_prop = R::rgamma(a, b) + pow(10,-323);
  double a_prop = pow(kappa_prop,2) / sigma_kappa_sq;
  
  double l = 0;
  double l_prop = 0;
  double K_n;
  double temp;
  for(int n=0; n<N; n++){
    K_n = sum(F.col(n));
    l += K_n*(lgamma(gamma*K_n+kappa) - lgamma(gamma+kappa));
    l_prop += K_n*(lgamma(gamma*K_n+kappa_prop) - lgamma(gamma+kappa_prop));
    for(int k=0; k<K_cur; k++){
      if(F(k,n)==1){
        temp = log(eta(k,k,n) / sum(eta.slice(n).row(k).t() % F.col(n)));
        l += (gamma+kappa-1) * temp;
        l_prop += (gamma+kappa_prop-1) * temp;
      }
    }
  }
  
  double MH_ratio = l_prop - l + (a_kappa+a_prop-a)*log(kappa_prop) - 
    (a_kappa+a-a_prop)*log(kappa) - (kappa_prop-kappa)*b_kappa +
    lgamma(a) - lgamma(a_prop) + (a-a_prop)*log(sigma_kappa_sq);
  
  if(log(randu()) < MH_ratio){
    kappa = kappa_prop;
  }
  
  return kappa;
}


//Function to find deterministic HM parameters for birth-death steps
// [[Rcpp::export]]
List DeterministicHMMParams(const mat& Y_, const mat& Y_lags, const vec& vecZ, 
                            int D, int r, const mat& K_mat, const mat& S_0, 
                            int n_0, int K_i, uvec avail, vec logdet_S_gb, 
                            vec logdet_S_bb, bool get_dets){
  //get_dets is a boolean indicating if new logdet_S_gb and logdet_S_bb should be returned
  
  cube A(D, D*r, K_i);
  cube Sigma(D, D, K_i);
  int n_k;
  int k_temp;
  double sign;
  List out4(4);
  
  for(int k=0; k<K_i; k++){
    k_temp = avail(k);
    uvec inK = find(vecZ == k_temp);
    n_k = inK.n_elem ;
    out4 = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
    if(n_k<2){n_k = 2;}
    Sigma.slice(k) = (as<mat>(out4("S_gb"))+S_0)/(n_k + n_0 - D - 1);
    A.slice(k) = as<mat>(out4("S_b")) * as<mat>(out4("S_bb_inv"));
    if(get_dets == true){
      log_det(logdet_S_gb(k_temp),sign,as<mat>(out4("S_gb")));
      log_det(logdet_S_bb(k_temp),sign,as<mat>(out4("S_bb")));
    }
  }
  
  return List::create(
      _["Sigma"] = Sigma,
      _["A"] = A,
      _["logdet_S_gb_prop"] = logdet_S_gb,
      _["logdet_S_bb_prop"] = logdet_S_bb
  );
}


//Function to get difference of log likelihood of Z and Z_prop for B-D steps
// [[Rcpp::export]]
double Lik_Z_Z_prop(const mat& TransMat_i, const mat& TransMat_new, double gamma,
                    double kappa, const vec& f_i, const vec& f_i_prop, int K_cur, 
                    int K_cur_new, const mat& lambda){
  
  int K_i_prop = sum(f_i_prop);
  int K_i = sum(f_i);
  double log_Lik = K_i*((K_i-1)*lgamma(gamma)+lgamma(gamma+kappa) - 
                        lgamma((K_i-1)*gamma+gamma+kappa)) -
                   K_i_prop*((K_i_prop-1)*lgamma(gamma)+lgamma(gamma+kappa) - 
                             lgamma((K_i_prop-1)*gamma+gamma+kappa));
  
  mat TM_i = TransMat_i + lambda.submat(0,0,K_cur-1,K_cur-1);
  mat TM_new = TransMat_new + lambda.submat(0,0,TransMat_new.n_rows-1,TransMat_new.n_cols-1);
  
  //account for likelihood of old Z
  for(int k=0; k<K_cur; k++){
    if(f_i(k) == 1){
      log_Lik += sum(lgamma(TM_i.row(k))) - lgamma(sum(TM_i.row(k)));
    }
  }
  //account for likelihood of new Z
  for(int k=0; k<K_cur_new; k++){
    if(f_i_prop(k) == 1){
      log_Lik -= sum(lgamma(TM_new.row(k))) - lgamma(sum(TM_new.row(k)));
    }
  }
  
  //return difference in log probs of Z_prop and Z
  return log_Lik;
}


//Function to calculate marginal likelihood of data in a state
// [[Rcpp::export]]
double MargY(int n_k, int D, int n_0, double logdet_S_gb, double logdet_S_bb, 
             double Marg_Prior){
  
  double out;
  if(logdet_S_gb > -pow(10,-400)){
    out = Marg_Prior - n_k*D/2*log2pi + logMultiGamma((n_k+n_0)/2,D) - 
          (n_k+n_0)/2*logdet_S_gb - logdet_S_bb/2;
  }else{
    if(n_k == 1){
      out = Marg_Prior - D/2*log2pi + logMultiGamma((1+n_0)/2,D);
    }else{
      out = 0;
    }
  }
  
  return out;
}


//Function to sample from BP-AR-HMM birth process for a single time series
// [[Rcpp::export]]
List Birth(const mat& Y_, const mat& Y_lags, vec f_i, const mat& Z, int W_0, int W_1, 
           int K_cur, int T, int i, int D, int r, const mat& K_mat, double gamma, 
           double kappa, const mat& S_0, int n_0, const vec& logdet_S_gb, 
           const vec& logdet_S_bb, const vec& f_i_new){
  
  //propose new feature vector with 1 additional feature
  int K_i = sum(f_i_new);
  uvec avail_new = find(f_i_new==1);
  
  //detrministically calculate new HMM parameters
  mat Z_bar = Z;
  Z_bar.submat(W_0,i,W_1,i).fill(K_cur);
  
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_bar), D, r, K_mat, 
                                       S_0, n_0, K_i, avail_new, empty, empty, false);
  cube A_new = as<cube>(outDHP("A"));
  cube Sigma_new = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur+1,K_cur+1);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  //calculate likelihoods under these new parameters
  mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new, Sigma_new, i, f_i_new, K_cur+1, T, D);
  
  //sample new state sequence given new feature vector and deterministic parameters
  List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, f_i_new, T);
  mat Z_new = Z;
  Z_new.col(i) = as<vec>(outZ("Z_new"));
  double Q_z_fwd = as<double>(outZ("log_prob"));
  mat TransMat_new = as<mat>(outZ("TransMat"));
  vec n_k_i_new = as<vec>(outZ("n_k_new"));
  vec vecZ = vectorise(Z_new);
  double sign;
  vec logdet_S_gb_prop(K_cur+1);
  logdet_S_gb_prop.head(K_cur) = logdet_S_gb;
  vec logdet_S_bb_prop(K_cur+1);
  logdet_S_bb_prop.head(K_cur) = logdet_S_bb;
  uvec inK = find(vecZ == K_cur);
  List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
  log_det(logdet_S_gb_prop(K_cur),sign,as<mat>(outS("S_gb")));
  log_det(logdet_S_bb_prop(K_cur),sign,as<mat>(outS("S_bb")));
  
  //calculate probability of reverse step
  List outDHP2 = DeterministicHMMParams(Y_, Y_lags, vecZ, D, r, K_mat, 
                                        S_0, n_0, K_i-1, avail_new.head(K_i-1),
                                        logdet_S_gb_prop, logdet_S_bb_prop, true);
  cube A_rev = as<cube>(outDHP2("A"));
  cube Sigma_rev = as<cube>(outDHP2("Sigma"));
  logdet_S_gb_prop = as<vec>(outDHP2("logdet_S_gb_prop"));
  logdet_S_bb_prop = as<vec>(outDHP2("logdet_S_bb_prop"));
  log_Likeli.rows(0,K_cur-1) = getLikeliMat(Y_, Y_lags, A_rev, Sigma_rev, i, f_i, K_cur, T, D);
  
  double Q_z_rev = Sample_Z_prob(log_Likeli.rows(0,K_cur-1), eta_null.submat(0,0,K_cur-1,K_cur-1), 
                                 K_cur, f_i, T, Z.col(i));
  
  return List::create(
    _["Q_z"] = (Q_z_rev - Q_z_fwd),
    _["z_i_new"] = Z_new.col(i),
    _["TransMat_new"] = TransMat_new,
    _["n_k_i_new"] = n_k_i_new,
    _["logdet_S_gb_prop"] = logdet_S_gb_prop,
    _["logdet_S_bb_prop"] = logdet_S_bb_prop
  );
}


//Function to sample from BP-AR-HMM death process for a single time series
// [[Rcpp::export]]
List Death(const mat& Y_, const mat& Y_lags, vec f_i, const mat& Z, int W_0, int W_1, 
           int K_cur, int T,int i, int D, int r, const mat& K_mat, 
           double gamma, double kappa, int k_remove, const mat& S_0, int n_0,
           const vec& logdet_S_gb, const vec& logdet_S_bb, const vec& f_i_new){
  
  //propose new feature vector f_i(k_remove) = 0
  int K_i = sum(f_i_new);
  uvec avail_new = find(f_i_new==1);
  uvec avail = find(f_i==1);
  
  //deterministically calculate HMM parameters
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z), D, r, K_mat, 
                                       S_0, n_0, K_i, avail_new, empty, 
                                       empty, false);
  cube A_new = as<cube>(outDHP("A"));
  cube Sigma_new = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur,K_cur);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  //find likelihoods given deterministic HMM parameters
  mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new, Sigma_new, i, f_i_new, K_cur, T, D);
  
  //propose new state sequence given new feature vector and deterministic HMM parameters
  List outZ = Sample_Z(log_Likeli, eta_null, K_cur, f_i_new, T);
  mat Z_new = Z;
  Z_new.col(i) = as<vec>(outZ("Z_new"));
  double Q_z_fwd = as<double>(outZ("log_prob"));
  mat TransMat_new = as<mat>(outZ("TransMat"));
  vec n_k_i_new = as<vec>(outZ("n_k_new"));
  
  //calculate log determinants of sufficient statistics for proposed state sequences
  vec logdet_S_gb_prop = logdet_S_gb;
  vec logdet_S_bb_prop = logdet_S_bb;
  vec vecZ = vectorise(Z_new);
  double sign;
  for(int k=0; k<K_cur; k++){
    if(f_i_new(k) == 1){
      uvec inK = find(vecZ == k);
      List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(outS("S_bb")));
    }
  }
  
  //calculate probability of reverse move
  mat Z_bar = Z;
  Z_bar.submat(W_0,i,W_1,i).fill(k_remove);
  List outDHP2 = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_bar), D, r, K_mat,
                                        S_0, n_0, K_i+1, avail, empty, empty, false);
  cube A_rev = as<cube>(outDHP2("A"));
  cube Sigma_rev = as<cube>(outDHP2("Sigma"));
  log_Likeli = getLikeliMat(Y_, Y_lags, A_rev, Sigma_rev, i, f_i, K_cur, T, D);
  double Q_z_rev = Sample_Z_prob(log_Likeli, eta_null, K_cur, f_i, T, Z.col(i));
  
  return List::create(
    _["Q_z"] = (Q_z_rev - Q_z_fwd),
    _["z_i_new"] = Z_new.col(i),
    _["TransMat_new"] = TransMat_new,
    _["n_k_i_new"] = n_k_i_new,
    _["logdet_S_gb_prop"] = logdet_S_gb_prop,
    _["logdet_S_bb_prop"] = logdet_S_bb_prop
  );
}


//Function to sample from BP-AR-HMM birth-death process for a single time series
// [[Rcpp::export]]
List BirthDeath(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z, 
                const vec& m, int L_min, int L_max, int T, int K_cur, int i, 
                int r, int D, double kappa, double gamma, const mat& K_mat, 
                int N, const mat& TransMat_i, double Lik_base, 
                const vec& logdet_S_gb, const vec& logdet_S_bb, const mat& n_k_i,
                const vec& n_k, int n_0, const mat& S_0, double anneal_temp,
                const mat& lambda, double Marg_Prior){
  
  //find the features uniqe to time series i
  uvec Uniq = find(m == 1 && F.col(i)==1);
  int n_Uniq = Uniq.n_elem;
  
  //propose birth if no unique features, otherwise pick a birth w/ prob 0.5
  double p_birth;
  if(n_Uniq==0){p_birth=1;}else{if(sum(F.col(i))==1){p_birth=1;}else{p_birth=0.5;}}
  
  //randomly select the window of data for birth-death proposal
  int L = randi(distr_param(L_min,L_max));
  int W_0 = randi(distr_param(0,T-L-1));
  int W_1 = W_0 + L;
  
  int k_remove;
  double Q_f;
  List out6(6);
  double Lik;
  int K_temp;
  vec f_i_prop(K_cur+1);
  f_i_prop.head(K_cur) = F.col(i);
  f_i_prop(K_cur) = 0;
  mat n_k_i_prop(K_cur+1,N,fill::zeros);
  n_k_i_prop.head_rows(K_cur) = n_k_i;
  
  if(randu()<p_birth){
    //propose a birth and account for likelihood of f_i and f_i_prop
    f_i_prop(K_cur) = 1;
    K_temp = K_cur+1;
    Q_f = log(0.5/(n_Uniq+1)) - log(p_birth);
    out6 = Birth(Y_, Y_lags, F.col(i), Z, W_0, W_1, K_cur, T, i, D, r, K_mat, 
                 gamma, kappa, S_0, n_0, logdet_S_gb, logdet_S_bb, f_i_prop);
    Lik = Lik_base - log(n_Uniq+1);
    n_k_i_prop.col(i) = as<vec>(out6("n_k_i_new"));
  }else{  
    //propose a death of feature k_remove and account for likelihood of f_i and f_i_prop
    K_temp = K_cur;
    k_remove = Uniq(floor(n_Uniq*randu()));
    f_i_prop.shed_row(K_cur);
    f_i_prop(k_remove) = 0;
    Q_f = log(0.5) - log(0.5/n_Uniq);
    out6 = Death(Y_, Y_lags, F.col(i), Z, W_0, W_1, K_cur, T, i, D, r, K_mat, gamma, 
                 kappa, k_remove, S_0, n_0, logdet_S_gb, logdet_S_bb, f_i_prop);
    Lik = -Lik_base + log(n_Uniq);
    n_k_i_prop.shed_row(K_cur);
    n_k_i_prop.col(i) = as<vec>(out6("n_k_i_new"));
  }
  mat Z_prop = Z;
  Z_prop.col(i) = as<vec>(out6("z_i_new"));
  mat TransMat_prop = as<mat>(out6("TransMat_new"));
  vec n_k_prop = sum(n_k_i_prop,1);
  double Q_z = as<double>(out6("Q_z"));
  vec logdet_S_gb_prop = as<vec>(out6("logdet_S_gb_prop"));
  vec logdet_S_bb_prop = as<vec>(out6("logdet_S_bb_prop"));
  
  //account for likelihood of Z_i and Z_i_prop
  Lik += Lik_Z_Z_prop(TransMat_i, TransMat_prop, gamma, kappa, 
                      F.col(i), f_i_prop, K_cur, K_temp, lambda);
  
  //account for likelihood of Y|Z
  for(int k=0; k<K_cur; k++){
    if(F(k,i)==1){
      Lik -= MargY(n_k(k), D, n_0, logdet_S_gb(k), logdet_S_bb(k), Marg_Prior);
    }
  }
  
  //account for likelihood of Y|Z_prop
  for(int k=0; k<K_temp; k++){
    if(f_i_prop(k) == 1){
      Lik += MargY(n_k_prop(k), D, n_0, logdet_S_gb_prop(k), logdet_S_bb_prop(k), Marg_Prior);
    }
  }
  
  //determine if you should accept the proposal or not
  if(log(randu()) < (Lik + (Q_z + Q_f)*anneal_temp)){
    mat F_prop(K_cur,N, fill::zeros);
    int K_prop;
    if(K_temp == K_cur){
      K_prop = K_cur - 1;
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_remove){Z_prop(t,n) = Z_prop(t,n)-1;}
        }
      }
      F_prop = F;
      F_prop.shed_row(k_remove);
      n_k_prop.shed_row(k_remove);
      n_k_i_prop.shed_row(k_remove);
      logdet_S_gb_prop.shed_row(k_remove);
      logdet_S_bb_prop.shed_row(k_remove);
      TransMat_prop.shed_row(k_remove);
      TransMat_prop.shed_col(k_remove);
    }else{
      K_prop = K_cur + 1;
      F_prop.insert_rows(K_cur,1);
      F_prop.head_rows(K_cur) = F;
      F_prop(K_cur,i) = 1;
    }
    
    return List::create(
      _["accept"] = true,
      _["K"] = K_prop,
      _["k_remove"] = k_remove,
      _["logdet_S_gb"] = logdet_S_gb_prop,
      _["logdet_S_bb"] = logdet_S_bb_prop,
      _["Z"] = Z_prop,
      _["F"] = F_prop,
      _["n_k"] = n_k_prop,
      _["n_k_i"] = n_k_i_prop,
      _["TransMat"] = TransMat_prop
    );
  }else{
    return List::create(
      _["accept"] = false,
      _["K"] = K_cur,
      _["k_remove"] = K_cur,
      _["logdet_S_gb"] = logdet_S_gb,
      _["logdet_S_bb"] = logdet_S_bb,
      _["Z"] = Z,
      _["F"] = F,
      _["n_k"] = n_k,
      _["n_k_i"] = n_k_i,
      _["TransMat"] = TransMat_i
    );
  }
}


//Function to select features for a split/merge move
// [[Rcpp::export]]
List FeatureSelect(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z,
                   int i, int j, int K_cur, const vec& n_k, int D, int r, int n_0,
                   double Marg_Prior, const vec& logdet_S_gb, const vec& logdet_S_bb, 
                   const mat& K_mat, int k_i=-1, int k_j=-1){
  
  int K_i = sum(F.col(i));
  double q_k = -log(K_i);
  uvec avail = find(F.col(i) == 1);
  if(k_i == -1){k_i = avail(floor(randu()*K_i));}
  
  vec log_p(K_cur);
  log_p.fill(-pow(10,400));
  double det1;
  double det2;
  double sign;
  vec vecZ = vectorise(Z);
  uvec ink_i = find(vecZ == k_i);
  for(int k=0; k<K_cur; k++){
    if(F(k,j) == 1 && k != k_i){
      uvec inK = find(vecZ == k);
      uvec inK2 = join_cols(ink_i,inK);
      List outS = get_MNIW_SS(Y_.cols(inK2), Y_lags.cols(inK2), K_mat, D, r);
      log_det(det1,sign,as<mat>(outS("S_gb")));
      log_det(det2,sign,as<mat>(outS("S_bb")));
      log_p(k) = MargY(n_k(k) + n_k(k_i), D, n_0, det1, det2, Marg_Prior) -
                 MargY(n_k(k), D, n_0, logdet_S_gb(k), logdet_S_bb(k), Marg_Prior) -
                 MargY(n_k(k_i), D, n_0, logdet_S_gb(k_i), logdet_S_bb(k_i), Marg_Prior);
    }
  }
  
  vec probs = Multinom_prob_stable(log_p, K_cur, F.col(j));
  if(F(k_i,j) == 1){probs(k_i) = 2*sum(probs);}
  probs = probs/sum(probs);
  if(k_j == -1){k_j = Sample_Multinom(probs);}
  q_k += log(probs(k_j));
  
  return List::create(
    _["k_i"] = k_i,
    _["k_j"] = k_j,
    _["q_k"] = q_k
  );
}


//Function to propose merge step
// [[Rcpp::export]]
List Merge(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z,
           int i, int j, int k_i, int k_j, int K_cur, int N, int T,
           int D, int r, const mat& K_mat, const mat& S_0, int n_0,
           double gamma, double kappa, const mat& target_Z, 
           const cube& TransCube){
  
  bool just_prob;
  if(target_Z(0,0)>=0){just_prob = true;}else{just_prob = false;}
  cube TransCube_prop(K_cur+1,K_cur+1,N,fill::zeros);
  TransCube_prop.subcube(0,0,0,K_cur-1,K_cur-1,N-1) = TransCube;
  
  mat F_prop(K_cur+1, N);
  F_prop.head_rows(K_cur) = F;
  F_prop.row(K_cur).fill(0);
  vec Has_i_or_j(N, fill::zeros);
  for(int n=0; n<N; n++){
    if(F(k_i,n)==1 || F(k_j,n)==1){
      F_prop(K_cur,n) = 1;
      Has_i_or_j(n) = 1;
    }
  }
  F_prop.row(k_i).fill(0);
  F_prop.row(k_j).fill(0);
  //determine the set of time series with feature i or j
  uvec ActiveSet = find(Has_i_or_j == 1);
  int n_active = ActiveSet.n_elem;
  //determine the set of features the time series in the active set possess
  vec TimesActive = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeat = find(TimesActive > 0);
  
  mat Z_prop = Z;
  for(int t=0; t<T; t++){
    if(Z_prop(t,i)==k_i || Z_prop(t,i)==k_j){Z_prop(t,i) = K_cur;}
    if(Z_prop(t,j)==k_i || Z_prop(t,j)==k_j){Z_prop(t,j) = K_cur;}
  }
  
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_prop), D, r, K_mat,
                                       S_0, n_0, ActiveFeat.n_elem, ActiveFeat, 
                                       empty, empty, false);
  cube A_new(D, D*r, K_cur+1);
  A_new.slices(ActiveFeat) = as<cube>(outDHP("A"));
  cube Sigma_new(D, D, K_cur+1);
  Sigma_new.slices(ActiveFeat) = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur+1,K_cur+1);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  //propose new state sequences for all time series in active set (other than i and j)
  mat Y_hat(D, T);
  int n_temp;
  int n_k;
  int n_k_last;
  double q = 0;
  vec vecZ(T*N);
  for(int n=0; n<n_active; n++){
    n_temp = ActiveSet(n);
    if(n_temp !=i && n_temp != j){
      uvec K_inc = find(F_prop.col(n_temp)==1);
      mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc), Sigma_new.slices(K_inc), n_temp, 
                                    F_prop.col(n_temp), K_cur+1, T, D);
      
      if(just_prob == false){
        List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(n_temp), T);
        q += as<double>(outZ("log_prob"));
        Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
        TransCube_prop.slice(n_temp) = as<mat>(outZ("TransMat"));
        n_k_last = as<vec>(outZ("n_k_new"))(K_cur);
      }else{
        q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(n_temp), 
                           T, target_Z.col(n_temp));
        Z_prop.col(n_temp) = target_Z.col(n_temp);
        uvec inK_cur = find(target_Z.col(n_temp) == K_cur);
        n_k_last = inK_cur.n_elem;
      }
      
      //update new state sufficient statistics if you added new timepoints
      if(n_k_last > 0){
        vecZ = vectorise(Z_prop);
        uvec inK = find(vecZ == K_cur);
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
    }
  }
  
  //propose state sequence for time series i
  uvec K_inc_i = find(F_prop.col(i)==1);
  mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_i), Sigma_new.slices(K_inc_i),
                                i, F_prop.col(i), K_cur+1, T, D);
  if(just_prob == false){
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(i), T);
    q += as<double>(outZ("log_prob"));
    Z_prop.col(i) = as<vec>(outZ("Z_new"));
    TransCube_prop.slice(i) = as<mat>(outZ("TransMat"));
  }else{
    q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(i), T, 
                       target_Z.col(i));
    Z_prop.col(i) = target_Z.col(i);
  }
  
  //propose state sequence for time series j
  uvec K_inc_j = find(F_prop.col(j)==1);
  log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_j), Sigma_new.slices(K_inc_j),
                            j, F_prop.col(j), K_cur+1, T, D);
  if(just_prob == false){
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(j), T);
    q += as<double>(outZ("log_prob"));
    Z_prop.col(j) = as<vec>(outZ("Z_new"));
    TransCube_prop.slice(j) = as<mat>(outZ("TransMat"));
  }else{
    q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(j), T, 
                       target_Z.col(j));
    Z_prop.col(j) = target_Z.col(j);
  }
  
  if(just_prob == false){
    return List::create(
      _["Z_prop"] = Z_prop,
      _["F_prop"] = F_prop,
      _["q"] = q,
      _["TransCube"] = TransCube_prop,
      _["ActiveSet"] = ActiveSet
    );
  }else{
    return List::create(
      _["Z_prop"] = Z,
      _["F_prop"] = F,
      _["q"] = q,
      _["TransCube"] = TransCube,
      _["ActiveSet"] = ActiveSet
    );
  }
}


//Function to propose split step
// [[Rcpp::export]]
List Split(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z, int i, 
           int j, int k_i, int K_cur, int N, int T, int D, int r, 
           const mat& K_mat, const mat& S_0, int n_0, double gamma, double kappa,
           double c, const mat& target_Z, const mat& target_F, 
           const cube& TransCube){
  
  bool just_prob;
  if(target_Z(0,0)>=0){just_prob = true;}else{just_prob = false;}
  cube TransCube_prop(K_cur+2,K_cur+2,N,fill::zeros);
  TransCube_prop.subcube(0,0,0,K_cur-1,K_cur-1,N-1) = TransCube;
  
  mat F_prop(K_cur+2, N);
  F_prop.head_rows(K_cur) = F;
  F_prop.rows(K_cur,K_cur+1).fill(0);
  vec Has_i(N, fill::zeros);
  for(int n=0; n<N; n++){
    if(F(k_i,n)==1){
      Has_i(n) = 1;
    }
  }
  F_prop.row(k_i).fill(0);
  F_prop(K_cur,i) = 1;
  F_prop(K_cur+1,j) = 1;
  //determine the set of time series with feature i or j
  uvec ActiveSet = find(Has_i == 1);
  int n_active = ActiveSet.n_elem;
  //determine the set of features the time series in the active set possess
  vec TimesActive = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeat = find(TimesActive > 0);
  
  mat Z_prop = Z;
  for(int t=0; t<T; t++){
    if(Z_prop(t,i)==k_i){Z_prop(t,i) = K_cur;}
    if(Z_prop(t,j)==k_i){Z_prop(t,j) = K_cur+1;}
  }
  
  cube A_new(D, D*r, K_cur+2);
  cube Sigma_new(D, D, K_cur+2);
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_prop), D, r, K_mat,
                                       S_0, n_0, ActiveFeat.n_elem, ActiveFeat, 
                                       empty, empty, false);
  A_new.slices(ActiveFeat) = as<cube>(outDHP("A"));
  Sigma_new.slices(ActiveFeat) = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur+2,K_cur+2);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  int n_temp;
  int m_a = 1;
  int m_b = 1;
  int scanned = 2;
  double q_z = 0;
  double q_f = 0;
  //p_ first element is p_10, second is p_01, third is p_11
  vec log_p(3);
  vec f_probs(3);
  vec f_star(K_cur+2);
  mat pi_n(K_cur+2, K_cur+2);
  vec n_k_temp(K_cur+2);
  int n_k_a;
  int n_k_b;
  int n_k;
  int f_pick;
  double L;
  vec temp3(3,fill::ones);
  vec vecZ(N*T);
  vec F_prop_temp(K_cur+2, fill::ones);
  //sample features allocations and state sequences from proposed split
    //or calculate log_prob of splitting to target F and Z
  for(int n=0; n<n_active; n++){
    n_temp = ActiveSet(n);
    if(n_temp !=i && n_temp != j){
      //get likelihood of each available state at each time point for given time series
      F_prop_temp.head(K_cur) = F_prop.col(n_temp).head(K_cur);
      uvec K_inc = find(F_prop_temp==1);
      mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc), Sigma_new.slices(K_inc), n_temp, 
                                    F_prop_temp, K_cur+2, T, D);
      
      //calculate log(p_10)
      f_star = F_prop.col(n_temp);
      f_star(K_cur) = 1;
      L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
      log_p(0) = log(m_a/(scanned+c)*(scanned-m_b+c)/(scanned+c)) + L;
      //calculate log(p_11)
      f_star(K_cur+1) = 1;
      L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
      log_p(2) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
      //calculate log(p_01)
      f_star(K_cur) = 0;
      L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
      log_p(1) = log(m_b/(scanned+c)*(scanned-m_a+c)/(scanned+c)) + L;
      f_probs = Multinom_prob_stable(log_p, 3, temp3);
      if(just_prob == false){
        //if not target allocation then sample feature allocations and get log_probs
        f_pick = Sample_Multinom(f_probs);
        if(f_pick == 0){
          F_prop(K_cur,n_temp) = 1;
          m_a++;
          q_f += log(f_probs(0));
        }
        if(f_pick == 1){
          F_prop(K_cur+1,n_temp) = 1;
          m_b++;
          q_f += log(f_probs(1));
        }
        if(f_pick == 2){
          F_prop(K_cur,n_temp) = 1;
          F_prop(K_cur+1,n_temp) = 1;
          m_a++;
          m_b++;
          q_f += log(f_probs(2));
        }
      }else{
        //or get log probability of target feature allocations
        m_a += target_F(K_cur,n_temp);
        m_b += target_F(K_cur+1,n_temp);
        if(target_F(K_cur,n_temp)==1 && target_F(K_cur+1,n_temp)==1){
          q_f += log(f_probs(2));
        }else{
          if(target_F(K_cur,n_temp)==1){
            q_f += log(f_probs(0));
          }
          if(target_F(K_cur+1,n_temp)==1){
            q_f += log(f_probs(1));
          }
        }
        F_prop.col(n_temp) = target_F.col(n_temp);
      }
      scanned++;
      
      //sample new state sequences given new feature allocations
      if(just_prob == false){
        List outZ = Sample_Z(log_Likeli, eta_null, K_cur+2, F_prop.col(n_temp), T);
        q_z += as<double>(outZ("log_prob"));
        Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
        TransCube_prop.slice(n_temp) = as<mat>(outZ("TransMat"));
        n_k_temp = as<vec>(outZ("n_k_new"));
        n_k_a = n_k_temp(K_cur);
        n_k_b = n_k_temp(K_cur+1);
      }else{
        q_z += Sample_Z_prob(log_Likeli, eta_null, K_cur+2, F_prop.col(n_temp), 
                             T, target_Z.col(n_temp));
        Z_prop.col(n_temp) = target_Z.col(n_temp);
        uvec inK_cur = find(target_Z.col(n_temp) == K_cur);
        n_k_a = inK_cur.n_elem;
        uvec inK_cur1 = find(target_Z.col(n_temp) == K_cur+1);
        n_k_b = inK_cur1.n_elem;
      }
      
      vecZ = vectorise(Z_prop);
      //update new state sufficient statistics if you added new timepoints
      if(n_k_a > 0){
        uvec inK = find(vecZ == K_cur);
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
      if(n_k_b > 0){
        uvec inK = find(vecZ == (K_cur+1));
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur+1) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur+1) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
    }
  }
  
  log_p.shed_row(0);
  f_probs.shed_row(0);
  //sample new feature vector and state sequence for time series i
  F_prop_temp.head(K_cur) = F_prop.col(i).head(K_cur);
  uvec K_inc_i = find(F_prop_temp==1);
  mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_i), Sigma_new.slices(K_inc_i), 
                                i, F_prop_temp, K_cur+2, T, D);
  
  //calculate log(p_10)
  f_star = F_prop.col(i);
  L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
  log_p(0) = log(m_a/(scanned+c)*(scanned-m_b+c)/(scanned+c)) + L;
  //calculate log(p_11)
  f_star(K_cur+1) = 1;
  L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
  log_p(1) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
  f_probs = Multinom_prob_stable(log_p, 2, temp3.head(2));
  if(just_prob == false){
    //if not target allocation then sample feature allocations and get log_probs
    f_pick = Sample_Multinom(f_probs);
    if(f_pick == 0){
      q_f += log(f_probs(0));
    }
    if(f_pick == 1){
      F_prop(K_cur+1,i) = 1;
      q_f += log(f_probs(1));
    }
  }else{
    //or get log probability of target feature allocations
    if(target_F(K_cur+1,i)==1){
      q_f += log(f_probs(1));
    }else{
      q_f += log(f_probs(0));
    }
    F_prop.col(i) = target_F.col(i);
  }
  
  if(just_prob == false){
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur+2, F_prop.col(i), T);
    q_z += as<double>(outZ("log_prob"));
    Z_prop.col(i) = as<vec>(outZ("Z_new"));
    TransCube_prop.slice(i) = as<mat>(outZ("TransMat"));
  }else{
    q_z += Sample_Z_prob(log_Likeli, eta_null, K_cur+2, F_prop.col(i), 
                         T, target_Z.col(i));
    Z_prop.col(i) = target_Z.col(i);
  }
  
  //sample new feature vector and state sequence for time series j
  F_prop_temp.head(K_cur) = F_prop.col(j).head(K_cur);
  uvec K_inc_j = find(F_prop_temp==1);
  log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_j), Sigma_new.slices(K_inc_j), 
                            j, F_prop_temp, K_cur+2, T, D);
    
  //calculate log(p_01)
  f_star = F_prop.col(j);
  L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
  log_p(0) = log(m_b/(scanned+c)*(scanned-m_a+c)/(scanned+c)) + L;
  //calculate log(p_11)
  f_star(K_cur) = 1;
  L = get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli);
  log_p(1) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
  f_probs = Multinom_prob_stable(log_p, 2, temp3.head(2));
  if(just_prob == false){
    //if not target allocation then sample feature allocations and get log_probs
    f_pick = Sample_Multinom(f_probs);
    if(f_pick == 0){
      q_f += log(f_probs(0));
    }
    if(f_pick == 1){
      F_prop(K_cur,j) = 1;
      q_f += log(f_probs(1));
    }
  }else{
    //or get log probability of target feature allocations
    if(target_F(K_cur,j)==1){
      q_f += log(f_probs(1));
    }else{
      q_f += log(f_probs(0));
    }
    F_prop.col(j) = target_F.col(j);
  }
  
  if(just_prob == false){
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur+2, F_prop.col(j), T);
    q_z += as<double>(outZ("log_prob"));
    Z_prop.col(j) = as<vec>(outZ("Z_new"));
    TransCube_prop.slice(j) = as<mat>(outZ("TransMat"));
  }else{
    q_z += Sample_Z_prob(log_Likeli, eta_null, K_cur+2, F_prop.col(j), 
                         T, target_Z.col(j));
    Z_prop.col(j) = target_Z.col(j);
  }
  
  if(just_prob == false){
    return List::create(
      _["Z_prop"] = Z_prop,
      _["F_prop"] = F_prop,
      _["q"] = q_z + q_f,
      _["TransCube"] = TransCube_prop,
      _["ActiveSet"] = ActiveSet
    );
  }else{
    return List::create(
      _["Z_prop"] = Z,
      _["F_prop"] = F,
      _["q"] = q_z + q_f,
      _["TransCube"] = TransCube,
      _["ActiveSet"] = ActiveSet
    );
  }
}


//Function to sample from BP-AR-HMM split-merge process
// [[Rcpp::export]]
List SplitMerge(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z,
                double kappa, double gamma, double alpha, double c, int N, int T,
                int K_cur, vec n_k, int D, int r, const mat& S_0, int n_0, 
                double Marg_Prior, const vec& logdet_S_gb, const vec& logdet_S_bb, 
                const mat& K_mat, const cube& TransCube, const mat& lambda,
                double anneal_temp){
  
  //randomly select time series to select features from
  int i = floor(randu()*N);
  int j = i;
  while(j == i){j = floor(randu()*N);}
  
  //select features for split or merge
  List out3 = FeatureSelect(Y_, Y_lags, F, Z, i, j, K_cur, n_k, D, r, n_0, 
                            Marg_Prior, logdet_S_gb, logdet_S_bb, K_mat);
  int k_i = as<int>(out3("k_i"));
  int k_j = as<int>(out3("k_j"));
  double q_k_fwd = as<double>(out3("q_k"));
  
  mat F_prop(K_cur+2, N);
  mat Z_prop(T, N);
  cube TransCube_prop(K_cur+2, K_cur+2, N);
  vec vecZ(T*N);
  vec n_k_prop(K_cur+2);
  vec logdet_S_gb_prop(K_cur+2);
  vec logdet_S_bb_prop(K_cur+2);
  double q_fwd;
  double q_rev;
  double q_k_rev;
  double sign;
  mat no_targ(1,1);
  no_targ.fill(-1); 
  mat target_Z = Z;
  uvec ActiveSet;
  bool merge_ind;
  if(k_i == k_j){
    //if k_i = k_j attempt a split move
    merge_ind = false;
    List out5 = Split(Y_, Y_lags, F, Z, i, j, k_i, K_cur, N, T, D, r, K_mat, S_0, 
                      n_0, gamma, kappa, c, no_targ, no_targ, TransCube);
    F_prop = as<mat>(out5("F_prop"));
    Z_prop = as<mat>(out5("Z_prop"));
    TransCube_prop = as<cube>(out5("TransCube"));
    q_fwd = as<double>(out5("q"));
    ActiveSet = as<uvec>(out5("ActiveSet"));
    vecZ = vectorise(Z_prop);
    for(int k=0; k<(K_cur+2); k++){
      uvec inK = find(vecZ==k);
      n_k_prop(k) = inK.n_elem;
      List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(outS("S_bb")));
    }
    out3 = FeatureSelect(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur+2, n_k_prop,
                         D, r, n_0, Marg_Prior, logdet_S_gb_prop, 
                         logdet_S_bb_prop, K_mat, K_cur, K_cur+1);
    q_k_rev = as<double>(out3("q_k"));
    for(int n=0; n<N; n++){
      for(int t=0; t<T; t++){
        if(target_Z(t,n) == k_i){target_Z(t,n) = K_cur+2;}
      }
    }
    List out5rev = Merge(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur, K_cur+1, K_cur+2, N,
                         T, D, r, K_mat, S_0, n_0, gamma, kappa, target_Z, TransCube_prop);
    q_rev = as<double>(out5rev("q"));
  }else{
    //otherwise attempt a merge move
    merge_ind = true;
    n_k_prop.shed_row(0);
    logdet_S_gb_prop.shed_row(0);
    logdet_S_bb_prop.shed_row(0);
    F_prop.shed_col(0);
    F_prop.shed_row(0);
    TransCube_prop.shed_col(0);
    TransCube_prop.shed_row(0);
    List out5 = Merge(Y_, Y_lags, F, Z, i, j, k_i, k_j, K_cur, N, T, D, r, K_mat, 
                      S_0, n_0, gamma, kappa, no_targ, TransCube);
    F_prop = as<mat>(out5("F_prop"));
    Z_prop = as<mat>(out5("Z_prop"));
    TransCube_prop = as<cube>(out5("TransCube"));
    q_fwd = as<double>(out5("q"));
    ActiveSet = as<uvec>(out5("ActiveSet"));
    vecZ = vectorise(Z_prop);
    for(int k=0; k<(K_cur+1); k++){
      uvec inK = find(vecZ==k);
      n_k_prop(k) = inK.n_elem;
      List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(outS("S_bb")));
    }
    out3 = FeatureSelect(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur+1, n_k_prop,
                         D, r, n_0, Marg_Prior, logdet_S_gb_prop, 
                         logdet_S_bb_prop, K_mat, K_cur, K_cur);
    q_k_rev = as<double>(out3("q_k"));   
    for(int n=0; n<N; n++){
      for(int t=0; t<T; t++){
        if(target_Z(t,n) == k_i){target_Z(t,n) = K_cur+1;}
        if(target_Z(t,n) == k_j){target_Z(t,n) = K_cur+2;}
      }
    }
    mat target_F(K_cur+3,N,fill::zeros);
    target_F.head_rows(K_cur) = F;
    target_F.row(k_i).fill(0);
    target_F.row(k_j).fill(0);
    target_F.row(K_cur+1) = F.row(k_i);
    target_F.row(K_cur+2) = F.row(k_j);
    List out5rev = Split(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur, K_cur+1, N, T, D, r, 
                         K_mat, S_0, n_0, gamma, kappa, c, target_Z, target_F, 
                         TransCube_prop);
    q_rev = as<double>(out5rev("q"));
  }
  
  vec m_Prop = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeatProp = find(m_Prop > 0);
  int n_Feat_active_prop = ActiveFeatProp.n_elem;
  vec m_cur = sum(F.cols(ActiveSet),1);
  uvec ActiveFeat = find(m_cur > 0);
  int n_Feat_active = ActiveFeat.n_elem;
  int n_active = ActiveSet.n_elem;
  double Lik;
  
  //Account for likelihood of F and F_prop
  if(merge_ind == false){
    Lik = log(alpha) + log(c) - lgamma(N+c) + 
          lgamma(m_Prop(K_cur)) + lgamma(N-m_Prop(K_cur)+c) +
          lgamma(m_Prop(K_cur+1)) + lgamma(N-m_Prop(K_cur+1)+c) -
          lgamma(m_cur(k_i)) - lgamma(N-m_cur(k_i)+c);
  }else{
    Lik = -log(alpha) - log(c) + lgamma(N+c) + 
          lgamma(m_cur(k_i)) + lgamma(N-m_cur(k_i)+c) +
          lgamma(m_cur(k_j)) + lgamma(N-m_cur(k_j)+c) -
          lgamma(m_Prop(K_cur)) - lgamma(N-m_Prop(K_cur)+c);
  }
  
  int n_hist;
  vec check(n_Feat_active, fill::zeros);
  for(int k=0; k<(n_Feat_active-1); k++){
    if(check(k)==0){
      n_hist = 1;
      for(int k2=(k+1); k<n_Feat_active; k++){
        if(check(k2)==0 && sum(abs(F.row(ActiveFeat(k2)) - F.row(ActiveFeat(k)))) == 0){
          n_hist++;
          check(k2) = 1;
        }
      }
      Lik += log(factorial(n_hist));
    }
  }
  
  vec check_prop(n_Feat_active_prop, fill::zeros);
  for(int k=0; k<(n_Feat_active_prop-1); k++){
    if(check_prop(k)==0){
      n_hist = 1;
      for(int k2=(k+1); k<n_Feat_active_prop; k++){
        if(check_prop(k2)==0 && sum(abs(F_prop.row(ActiveFeatProp(k2)) - F_prop.row(ActiveFeatProp(k)))) == 0){
          n_hist++;
          check_prop(k2) = 1;
        }
      }
      Lik -= log(factorial(n_hist));
    }
  }
  
  //account for likelihood of Z and Z_prop
  int n_temp;
  int K_temp = F_prop.n_rows;
  for(int n=0; n<n_active; n++){
    n_temp = ActiveSet(n);
    Lik += Lik_Z_Z_prop(TransCube.slice(n_temp), TransCube_prop.slice(n_temp), 
                        gamma, kappa, F.col(n_temp), F_prop.col(n_temp), 
                        K_cur, K_temp, lambda);
  }
  
  //account for likelihood of Y|Z and Y_Z_prop
  int k_temp1;
  int k_temp2;
  int int_temp;
  if(merge_ind==true){
    int_temp = n_Feat_active_prop;
    k_temp2 = ActiveFeat(n_Feat_active-1);
    Lik -= MargY(n_k(k_temp2), D, n_0, logdet_S_gb(k_temp2), 
                 logdet_S_bb(k_temp2), Marg_Prior);
  }else{
    int_temp = n_Feat_active;
    k_temp1 = ActiveFeatProp(n_Feat_active_prop-1);
    Lik += MargY(n_k_prop(k_temp1), D, n_0, logdet_S_gb_prop(k_temp1), 
                 logdet_S_bb_prop(k_temp1), Marg_Prior);    
  }
  for(int k=0; k<int_temp ; k++){
    k_temp1 = ActiveFeatProp(k);
    k_temp2 = ActiveFeat(k);
    Lik += MargY(n_k_prop(k_temp1), D, n_0, logdet_S_gb_prop(k_temp1), 
                 logdet_S_bb_prop(k_temp1), Marg_Prior) - 
           MargY(n_k(k_temp2), D, n_0, logdet_S_gb(k_temp2), 
                 logdet_S_bb(k_temp2), Marg_Prior);
  }
  
  //determine if you should accept the proposal or not
  if(log(randu()) < (Lik + (q_rev - q_fwd + q_k_rev - q_k_fwd)*anneal_temp)){
    if(merge_ind==false){
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_i){Z_prop(t,n) = Z_prop(t,n)-1;}
        }
      }
      F_prop.shed_row(k_i);
      n_k_prop.shed_row(k_i);
      logdet_S_gb_prop.shed_row(k_i);
      logdet_S_bb_prop.shed_row(k_i);
      TransCube_prop.shed_row(k_i);
      TransCube_prop.shed_col(k_i);
    }else{
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_i && Z_prop(t,n)>k_j){
            Z_prop(t,n) = Z_prop(t,n)-2;
          }else{
            if(Z_prop(t,n)>k_i || Z_prop(t,n)>k_j){
              Z_prop(t,n) = Z_prop(t,n)-1;
            }
          }
        }
      }
      if(k_i > k_j){
        F_prop.shed_row(k_i);
        F_prop.shed_row(k_j);
        n_k_prop.shed_row(k_i);
        n_k_prop.shed_row(k_j);
        logdet_S_gb_prop.shed_row(k_i);
        logdet_S_gb_prop.shed_row(k_j);
        logdet_S_bb_prop.shed_row(k_i);
        logdet_S_bb_prop.shed_row(k_j);
        TransCube_prop.shed_row(k_i);
        TransCube_prop.shed_col(k_i);
        TransCube_prop.shed_row(k_j);
        TransCube_prop.shed_col(k_j);
      }else{
        F_prop.shed_row(k_j);
        F_prop.shed_row(k_i);
        n_k_prop.shed_row(k_j);
        n_k_prop.shed_row(k_i);
        logdet_S_gb_prop.shed_row(k_j);
        logdet_S_gb_prop.shed_row(k_i);
        logdet_S_bb_prop.shed_row(k_j);
        logdet_S_bb_prop.shed_row(k_i);
        TransCube_prop.shed_row(k_j);
        TransCube_prop.shed_col(k_j);
        TransCube_prop.shed_row(k_i);
        TransCube_prop.shed_col(k_i);
      }
    }
    return List::create(
      _["accept"] = true,
      _["merge_ind"] = merge_ind,
      _["logdet_S_gb"] = logdet_S_gb_prop,
      _["logdet_S_bb"] = logdet_S_bb_prop,
      _["Z"] = Z_prop,
      _["F"] = F_prop,
      _["n_k"] = n_k_prop,
      _["TransCube"] = TransCube_prop
    );
  }else{
    return List::create(
      _["accept"] = false,
      _["merge_ind"] = merge_ind,
      _["logdet_S_gb"] = logdet_S_gb,
      _["logdet_S_bb"] = logdet_S_bb,
      _["Z"] = Z,
      _["F"] = F,
      _["n_k"] = n_k,
      _["TransCube"] = TransCube
    );
  }
}




//Function to fit Sticky HDP VAR HMM model with MNIW conjugate prior
// [[Rcpp::export]]
List BP_AR_HMM(const cube& Y, int N_samples, int L_min, int L_max, int r=1, 
               double K_scale=1, int n_0=0, int K_init=1, int PrintEvery=100,
               int KeepEvery=1, int BDper=1, int SMper=1, double burnin=5000,
               int ProcessPer=1, bool reduce=true, int SMwarm=0,
               double c=1, double a_c=1, double b_c=1, double sigma_c_sq=1,
               double alpha=1, double a_alpha=1, double b_alpha=1,
               double gamma=1, double a_gamma=1, double b_gamma=1, double sigma_gamma_sq=1,
               double kappa=1, double a_kappa=1, double b_kappa=0.01, double sigma_kappa_sq=50){
  
  double T = Y.n_cols - r;
  int D = Y.n_rows;
  int N = Y.n_slices;
  if(n_0==0){n_0 = D+2;}
  
  //reshape Y into a DxT*N dimension matrix by concatenating slices and dropping first r time points of each series
  //and create corresponding design matrix of lagged values
  mat Y_(D,T*N);
  mat Y_lags(D*r,T*N);
  for(int n=0; n<N; n++){
    Y_.cols(n*T,(n+1)*T-1) = Y.slice(n).cols(r,T+r-1);
    for(int t=0; t<T; t++){
      Y_lags.submat(0, n*T+t, D*r-1, n*T+t) = vectorise(fliplr(Y.slice(n).cols(t, t+r-1)));
    }
  }
  
  //set Inverse Wishart prior hyperparameters to those suggested by Fox
  vec Y_bar(D);
  mat Sigma_Bar(D,D,fill::zeros);
  for(int n=0; n<N; n++){
    Y_bar = mean(Y.slice(n).cols(1,T+r-1)-Y.slice(n).cols(0,T+r-2),1);
    for(int t=1; t<(T+r); t++){
      Sigma_Bar += (Y.slice(n).col(t)-Y.slice(n).col(t-1)-Y_bar)*(Y.slice(n).col(t)-Y.slice(n).col(t-1)-Y_bar).t();
    }
  }
  mat S_0 = 0.75*Sigma_Bar/(T+r)/N;
  if(n_0 == 0){
    n_0 = D*r + 2;
  }
  mat K_mat = K_scale*eye(D*r, D*r);  
  //cache prior parts of marginal Y likelihood for later since it remains constant
  double det1;
  double det2;
  double sign;
  log_det(det1,sign,S_0);
  log_det(det2,sign,K_mat);
  double Marg_Prior = -logMultiGamma(n_0/2,D) + (n_0/2)*det1 + det2/2;
  
  int K_cur = K_init;
  vec K_hist(N_samples);
  mat lambda(K_cur+3,K_cur+3);
  lambda.fill(gamma);
  lambda.diag().fill(gamma+kappa);
  vec gamma_hist(N_samples);
  vec kappa_hist(N_samples);
  vec alpha_hist(N_samples);
  vec c_hist(N_samples);
  
  //initialize A and Sigma for first features to posterior draws from full population
  field<cube> A(N_samples);
  cube A_cur(D, D*r, K_cur);
  field<cube> Sigma(N_samples);
  cube Sigma_cur(D, D, K_cur);
  List outS = get_MNIW_SS(Y_, Y_lags, K_mat, D, r);
  for(int k=0; k<K_cur; k++){      
    List out_temp = Sample_MNIW(D, r, as<mat>(outS("S_gb")), as<mat>(outS("S_bb_inv")), 
                                as<mat>(outS("S_b")), S_0, n_0, floor(N*T/K_cur));
    A_cur.slice(k) = as<mat>(out_temp("A_new"));
    Sigma_cur.slice(k) = as<mat>(out_temp("Sigma_new"));
  }
  
  //inialize so that all time series have all initial features
  mat F(K_cur, N, fill::ones);
  vec m(K_cur);
  m = sum(F,1);
  
  //calculate likelihood of data in each available feature for each time series
  cube log_Likeli(K_cur, T, N);
  vec F_ones(K_cur, fill::ones);
  for(int n=0; n<N; n++){
    log_Likeli.slice(n) = getLikeliMat(Y_, Y_lags, A_cur, Sigma_cur, n, F_ones, K_cur, T, D);
  }
  
  //initiate transition distributions, eta, for each subject
  cube eta(K_cur, K_cur, N, fill::ones);
  
  //initiate state sequences and sufficient statistics structures 
  cube Z(T, N, N_samples);
  mat Z_cur(T, N);
  cube TransCube(K_cur, K_cur, N,fill::zeros);
  mat n_k_i(K_cur,N);
  vec logdet_S_gb(K_cur);
  vec logdet_S_bb(K_cur);
  
  //MCMC sampler with burn-in with annealing to improve mixing
  int n_sample=0;
  int n_thin=1;  
  int burn_iter=0;
  int warm_iter=0;
  double anneal_temp = (burn_iter + pow(10,-323))/burnin;
  double Lik_base = log(alpha) + log(c) + lgamma(N-1+c) - lgamma(N+c);
  while(n_sample < N_samples){
    //cout<<"A";
    //sample shared feature assignments
    for(int n=0; n<N; n++){
      F.col(n) = sample_F(F.col(n), K_cur, N, T, D, m-F.col(n), c, 
                          eta.slice(n), log_Likeli.slice(n));
      m = sum(F,1);
    }
    //cout<<"B";
    
    //sample state sequences
    TransCube.zeros(K_cur, K_cur, N);
    n_k_i.zeros(K_cur,N);
    for(int n=0; n<N; n++){
      List outZ = Sample_Z(log_Likeli.slice(n), eta.slice(n), K_cur, F.col(n), T);
      Z_cur.col(n) = as<vec>(outZ("Z_new"));
      TransCube.slice(n) += as<mat>(outZ("TransMat"));
      n_k_i.col(n) = as<vec>(outZ("n_k_new"));
    }
    vec n_k = sum(n_k_i, 1);
    vec vecZ = vectorise(Z_cur);
    for(int k=0; k<K_cur; k++){
      uvec inK = find(vecZ == k);
      outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb(k),sign,as<mat>(outS("S_bb")));
    }
    //cout<<"C";
    
    //propose birth/death step for each time series
    int K_old = K_cur;
    if(warm_iter >= SMwarm){
      for(int BD_iter=0; BD_iter<BDper; BD_iter++){
        for(int n=0; n<N; n++){
          List outBD = BirthDeath(Y_, Y_lags, F, Z_cur, m, L_min, L_max, T, K_cur, n,
                                  r, D, kappa, gamma, K_mat, N, TransCube.slice(n), 
                                  Lik_base, logdet_S_gb, logdet_S_bb, n_k_i, n_k, 
                                  n_0, S_0, anneal_temp, lambda, Marg_Prior);
          if(as<bool>(outBD("accept")) == true){
            if(as<int>(outBD("K")) > K_cur){
              //cout<<"Birth";
              TransCube.insert_rows(K_cur,1);
              TransCube.insert_cols(K_cur,1);
              lambda.insert_rows(K_cur,1);
              lambda.insert_cols(K_cur,1);
              lambda.fill(gamma);
              lambda.diag().fill(gamma+kappa);
              K_cur++;
            }else{
              //cout<<"Death";
              int k_remove = as<int>(outBD("k_remove"));
              TransCube.shed_row(k_remove);
              TransCube.shed_col(k_remove);
              lambda.shed_row(k_remove);
              lambda.shed_col(k_remove);
              K_cur = K_cur - 1;
            }
            F.set_size(K_cur,N);
            mat F_temp = as<mat>(outBD("F"));
            F = F_temp;
            m.set_size(K_cur);
            m = sum(F,1);
            logdet_S_gb.set_size(K_cur);
            vec logdet_S_gb_temp = as<vec>(outBD("logdet_S_gb"));
            logdet_S_gb = logdet_S_gb_temp;
            logdet_S_bb.set_size(K_cur);
            vec logdet_S_bb_temp = as<vec>(outBD("logdet_S_bb"));
            logdet_S_bb = logdet_S_bb_temp;
            n_k.set_size(K_cur);
            vec n_k_temp = as<vec>(outBD("n_k"));
            n_k = n_k_temp;
            n_k_i.set_size(K_cur,N);
            mat n_k_i_temp = as<mat>(outBD("n_k_i"));
            n_k_i = n_k_i_temp;
            mat Z_temp = as<mat>(outBD("Z"));
            Z_cur = Z_temp;
            mat TransMat_temp = as<mat>(outBD("TransMat"));
            TransCube.slice(n) = TransMat_temp;
          }
        }
      }
    }
    
    //cout<<"D";
    for(int SM_iter=0; SM_iter<SMper; SM_iter++){
      //propose SMper many split merge moves
      List outSM = SplitMerge(Y_, Y_lags, F, Z_cur, kappa, gamma, alpha, c, N, T,
                              K_cur, n_k, D, r, S_0, n_0, Marg_Prior, logdet_S_gb,
                              logdet_S_bb, K_mat, TransCube, lambda, anneal_temp);
      if(as<bool>(outSM("accept")) == true){
        if(as<bool>(outSM("merge_ind")) == false){
          //cout<<"Split";
          lambda.insert_rows(K_cur,1);
          lambda.insert_cols(K_cur,1);
          lambda.fill(gamma);
          lambda.diag().fill(gamma+kappa);
          K_cur++;
        }else{
          //cout<<"Merge";
          lambda.shed_row(0);
          lambda.shed_col(0);
          K_cur = K_cur - 1;
        }
        F.set_size(K_cur,N);
        mat F_temp = as<mat>(outSM("F"));
        F = F_temp;
        m.set_size(K_cur);
        m = sum(F,1);
        logdet_S_gb.set_size(K_cur);
        vec logdet_S_gb_temp = as<vec>(outSM("logdet_S_gb"));
        logdet_S_gb = logdet_S_gb_temp;
        logdet_S_bb.set_size(K_cur);
        vec logdet_S_bb_temp = as<vec>(outSM("logdet_S_bb"));
        logdet_S_bb = logdet_S_bb_temp;
        n_k.set_size(K_cur);
        vec n_k_temp = as<vec>(outSM("n_k"));
        n_k = n_k_temp;
        mat Z_temp = as<mat>(outSM("Z"));
        Z_cur = Z_temp;
        TransCube.set_size(K_cur, K_cur, N);
        cube TransCube_temp = as<cube>(outSM("TransCube"));
        TransCube = TransCube_temp;
      }
    }
    //delete or add to structures to match new K_cur from BD/SM steps
    if(K_cur != K_old){
      A_cur.set_size(D,D*r,K_cur);
      Sigma_cur.set_size(D,D,K_cur);
      log_Likeli.set_size(K_cur,T,N);
      eta.set_size(K_cur,K_cur,N);
    }
    //cout<<"E";
    
    //sample state emission parameters and calculate new likelihoods
    vecZ = vectorise(Z_cur);
    for(int k=0; k<K_cur; k++){
      uvec inK = find(vecZ == k);
      outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb(k),sign,as<mat>(outS("S_bb")));
      List out_temp = Sample_MNIW(D, r, as<mat>(outS("S_gb")), as<mat>(outS("S_bb_inv")), 
                                  as<mat>(outS("S_b")), S_0, n_0, n_k(k));
      A_cur.slice(k) = as<mat>(out_temp("A_new"));
      Sigma_cur.slice(k) = as<mat>(out_temp("Sigma_new"));
    }
    F_ones.ones(K_cur);
    for(int n=0; n<N; n++){
      log_Likeli.slice(n) = getLikeliMat(Y_, Y_lags, A_cur, Sigma_cur, n, F_ones, K_cur, T, D);
    }
    //cout<<"F";
    
    //sample state transition parameters
    for(int n=0; n<N; n++){
      eta.slice(n) = sample_eta(TransCube.slice(n), gamma, kappa, K_cur, sum(F.col(n)));
    }
    
    //sample hyperparameters, repeat many times to improve acceptance (or don't sample at all)
    for(int ProcIter=0; ProcIter<ProcessPer; ProcIter++){
      alpha = sample_alpha(a_alpha, b_alpha, c, K_cur, N);
      c = sample_c(c, a_c, b_c, alpha, sigma_c_sq, K_cur, N, m);
      gamma = sample_gamma(gamma, a_gamma, b_gamma, sigma_gamma_sq, N, K_cur, 
                           kappa, F, eta);
      kappa = sample_kappa(kappa, a_kappa, b_kappa, sigma_kappa_sq, N, K_cur, 
                           gamma, F, eta);
      lambda.fill(gamma);
      lambda.diag().fill(gamma+kappa);
      Lik_base = log(alpha) + log(c) + lgamma(N-1+c) - lgamma(N+c);
    }
    
    //Only save parameter values every KeepEvery iterations
    if(warm_iter < SMwarm){
      warm_iter++;
      if(warm_iter%PrintEvery == 0){cout << warm_iter << " Split-Merge warmup samples complete" << endl;}
    }else{
      if(burn_iter < burnin){
        burn_iter++;
        anneal_temp = (burn_iter+pow(10,-323))/burnin;
        if(burn_iter%PrintEvery == 0){cout << burn_iter << " burn-in samples complete" << endl;}
      }else{
        if(n_thin%KeepEvery==0){
          if(reduce == true){
            vec used0 = unique(Z_cur);
            uvec used = conv_to<uvec>::from(used0);
            cube A_out = A_cur.slices(used);
            cube Sigma_out = Sigma_cur.slices(used);
            mat Z_out(T,N);
            vec key(K_cur);
            key.fill(-1);
            for(int k=0; k<K_cur; k++){
              uvec temp_ind = find(used0 == k);
              if(temp_ind.n_elem>0){key(k) = temp_ind(0)+1;}
            }
            for(int n1=0; n1<T; n1++){
              for(int n2=0; n2<N; n2++){
                Z_out(n1,n2) = key(Z_cur(n1,n2));
              }
            }
            A(n_sample) = A_out;
            Sigma(n_sample) = Sigma_out;
            Z.slice(n_sample) = Z_out;
          }else{
            A(n_sample) = A_cur;
            Sigma(n_sample) = Sigma_cur;
            Z.slice(n_sample) = Z_cur;
          }
          K_hist(n_sample) = K_cur;
          gamma_hist(n_sample) = gamma;
          kappa_hist(n_sample) = kappa;
          alpha_hist(n_sample) = alpha;
          c_hist(n_sample) = c;
          n_sample++;
          n_thin=1;    
          if(n_sample%PrintEvery == 0){cout << n_sample<< " posterior samples complete" << endl;}
        }else{
          n_thin++;
        } 
      }
    }
  }
  
  return List::create(
    _["Sigma"] = Sigma,
    _["A"] = A,
    _["Z"] = Z,
    _["K"] = K_hist,
    _["gamma"] = gamma_hist,
    _["kappa"] = kappa_hist,
    _["alpha"] = alpha_hist,
    _["c"] = c_hist
  );
}




//Function to sample feature indicators for GBP-AR-HMM
// [[Rcpp::export]]
vec sample_F_group(vec f_i, int K_cur, int N_groups, int T, int D, const vec& m_i,
                   double c, const mat& eta_i, const cube& log_Likeli_i, int N_g){

  vec f_star(K_cur);
  vec Mes(K_cur);
  double L = 0;
  for(int n_g=0; n_g<N_g; n_g++){
    L += get_L_Y_given_f_theta(f_i, K_cur, T, eta_i, log_Likeli_i.slice(n_g));
  }

  for(int k=0; k<K_cur; k++){
    f_star = f_i;
    if(m_i(k)>0){
      if(f_i(k)==sum(f_i)){f_i(k)=1;}else{
        if(f_i(k) == 0){
          f_star(k) = 1;
        }else{
          f_star(k) = 0;
        }
        double P_on =  m_i(k) / (N_groups + m_i(k) + c);
        double P_star = log(f_star(k)*P_on + (1-f_star(k))*(1-P_on));
        double P =log(f_i(k)*P_on + (1-f_i(k))*(1-P_on));
        double L_star = 0;
        for(int n_g=0; n_g<N_g; n_g++){
          L_star += get_L_Y_given_f_theta(f_star, K_cur, T, eta_i, log_Likeli_i.slice(n_g));
        }
        if(log(randu()) < (P_star+L_star-P-L)){
          f_i = f_star;
          L = L_star;
        }
      }
    }
  }
  
  return f_i;
}


//Function to sample from GBP-AR-HMM birth process for a single group of time series group
// [[Rcpp::export]]
List BirthGroup(const mat& Y_, const mat& Y_lags, const vec& f_i, const mat& Z, int W_0, 
                int W_1, int K_cur, int T, const uvec& group_i, int D, int r, 
                const mat& K_mat, double gamma, double kappa, const mat& S_0, int n_0,
                const vec& logdet_S_gb, const vec& logdet_S_bb, int N_g, const vec& f_i_new){
  
  //propose new feature vector with 1 additional feature
  int K_i = sum(f_i_new);
  uvec avail_new = find(f_i_new==1);
  
  //detrministically calculate new HMM parameters
  mat Z_bar = Z;
  int n_temp = group_i(floor(randu()*N_g));
  Z_bar.submat(W_0,n_temp,W_1,n_temp).fill(K_cur);
  
  cube A_new(D,D*r,K_i);
  cube Sigma_new(D,D,K_i);
  mat eta_null(K_cur+1,K_cur+1);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_bar), D, r, K_mat, 
                                       S_0, n_0, K_i, avail_new, empty, empty, false);
  A_new = as<cube>(outDHP("A"));
  Sigma_new = as<cube>(outDHP("Sigma"));
  
  //calculate likelihoods under these new parameters
  double Q_z_fwd = 0;
  mat Z_new = Z;
  mat TransMat_new(K_cur+1,K_cur+1,fill::zeros);
  vec n_k_i_new(K_cur+1, fill::zeros);
  for(int i=0; i<N_g; i++){
    //sample new state sequence given new feature vector and deterministic parameters
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new, Sigma_new, group_i(i), f_i_new, K_cur+1, T, D);
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, f_i_new, T);
    Z_new.col(group_i(i)) = as<vec>(outZ("Z_new"));
    Q_z_fwd += as<double>(outZ("log_prob"));
    TransMat_new += as<mat>(outZ("TransMat"));
    n_k_i_new += as<vec>(outZ("n_k_new"));
  }
  
  vec vecZ = vectorise(Z_new);
  double sign;
  vec logdet_S_gb_prop(K_cur+1);
  logdet_S_gb_prop.head(K_cur) = logdet_S_gb;
  vec logdet_S_bb_prop(K_cur+1);
  logdet_S_bb_prop.head(K_cur) = logdet_S_bb;
  uvec inK = find(vecZ == K_cur);
  List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
  log_det(logdet_S_gb_prop(K_cur),sign,as<mat>(outS("S_gb")));
  log_det(logdet_S_bb_prop(K_cur),sign,as<mat>(outS("S_bb")));
  //calculate probability of reverse step
  cube A_rev(D,D*r,K_i-1);
  cube Sigma_rev(D,D,K_i-1);
  List outDHP2 = DeterministicHMMParams(Y_, Y_lags, vecZ, D, r, K_mat, S_0, 
                                        n_0, K_i-1, avail_new.head(K_i-1),
                                        logdet_S_gb_prop, logdet_S_bb_prop, true);
  A_rev = as<cube>(outDHP2("A"));
  Sigma_rev = as<cube>(outDHP2("Sigma"));
  logdet_S_gb_prop = as<vec>(outDHP2("logdet_S_gb_prop"));
  logdet_S_bb_prop = as<vec>(outDHP2("logdet_S_bb_prop"));
  
  double Q_z_rev = 0;
  for(int i=0; i<N_g; i++){
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_rev, Sigma_rev, group_i(i), f_i, K_cur, T, D);
    Q_z_rev += Sample_Z_prob(log_Likeli, eta_null.submat(0,0,K_cur-1,K_cur-1), 
                             K_cur, f_i, T, Z.col(group_i(i)));
  }
  
  return List::create(
    _["Q_z"] = (Q_z_rev - Q_z_fwd),
    _["Z_new"] = Z_new,
    _["TransMat_new"] = TransMat_new,
    _["n_k_i_new"] = n_k_i_new,
    _["logdet_S_gb_prop"] = logdet_S_gb_prop,
    _["logdet_S_bb_prop"] = logdet_S_bb_prop
  );
}


//Function to sample from GBP-AR-HMM death process for a single time series group
// [[Rcpp::export]]
List DeathGroup(const mat& Y_, const mat& Y_lags, const vec& f_i, const mat& Z, int W_0, 
                int W_1, int K_cur, int T, const uvec& group_i, int D, int r, 
                const mat& K_mat, double gamma, double kappa, int k_remove, const mat& S_0, 
                int n_0, const vec& logdet_S_gb, const vec& logdet_S_bb, int N_g, const vec& f_i_new){
  
  int K_i = sum(f_i_new);
  uvec avail_new = find(f_i_new==1);
  uvec avail = find(f_i==1);
  
  double Q_z_fwd = 0;
  mat Z_new = Z;
  mat TransMat_new(K_cur,K_cur,fill::zeros);
  vec n_k_i_new(K_cur, fill::zeros);
  
  //deterministically calculate HMM parameters
  cube A_new(D,D*r,K_i);
  cube Sigma_new(D,D,K_i);
  mat eta_null(K_cur,K_cur);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z), D, r, K_mat, 
                                       S_0, n_0, K_i, avail_new, empty, empty, false);
  A_new = as<cube>(outDHP("A"));
  Sigma_new = as<cube>(outDHP("Sigma"));
  
  //find likelihoods given deterministic HMM parameters
  for(int i=0; i<N_g; i++){
    //sample new state sequence given new feature vector and deterministic parameters
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new, Sigma_new, group_i(i), f_i_new, K_cur, T, D);
    List outZ = Sample_Z(log_Likeli, eta_null, K_cur, f_i_new, T);
    Z_new.col(group_i(i)) = as<vec>(outZ("Z_new"));
    Q_z_fwd += as<double>(outZ("log_prob"));
    TransMat_new += as<mat>(outZ("TransMat"));
    n_k_i_new += as<vec>(outZ("n_k_new"));
  }
  
  //calculate log determinants of sufficient statistics for proposed state sequences
  vec logdet_S_gb_prop(K_cur);
  logdet_S_gb_prop = logdet_S_gb;
  vec logdet_S_bb_prop(K_cur);
  logdet_S_bb_prop = logdet_S_bb;
  vec vecZ = vectorise(Z_new);
  double sign;
  for(int k=0; k<K_cur; k++){
    if(f_i_new(k) == 1){
      uvec inK = find(vecZ == k);
      List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(outS("S_bb")));
    }
  }
  
  //calculate probability of reverse move
  mat Z_bar = Z;
  int n_temp = group_i(floor(randu()*N_g));
  Z_bar.submat(W_0,n_temp,W_1,n_temp).fill(k_remove);
  cube A_rev(D,D*r,K_i+1);
  cube Sigma_rev(D,D,K_i+1);
  List outDHP2 = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_bar), D, r, K_mat,
                                        S_0, n_0, K_i+1, avail, empty, empty, false);
  A_rev = as<cube>(outDHP2("A"));
  Sigma_rev = as<cube>(outDHP2("Sigma"));
  double Q_z_rev = 0;
  for(int i=0; i<N_g; i++){
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_rev, Sigma_rev, group_i(i), f_i, K_cur, T, D);
    Q_z_rev += Sample_Z_prob(log_Likeli, eta_null, K_cur, f_i, T, Z.col(group_i(i)));
  }
  
  return List::create(
    _["Q_z"] = (Q_z_rev - Q_z_fwd),
    _["Z_new"] = Z_new,
    _["TransMat_new"] = TransMat_new,
    _["n_k_i_new"] = n_k_i_new,
    _["logdet_S_gb_prop"] = logdet_S_gb_prop,
    _["logdet_S_bb_prop"] = logdet_S_bb_prop
  );
}


//Function to sample from BP-AR-HMM birth-death process for a single time series
// [[Rcpp::export]]
List BirthDeathGroup(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z, 
                     const vec& m, int L_min, int L_max, int T, int K_cur, int i, 
                     int r, int D, double kappa, double gamma, const mat& K_mat, 
                     int N_groups, const mat& TransMat_i, double Lik_base, 
                     const vec& logdet_S_gb, const vec& logdet_S_bb, const mat& n_k_i, 
                     const vec& n_k, int n_0, const mat& S_0, double anneal_temp, 
                     const mat& lambda, double Marg_Prior, const uvec& group_i, 
                     int N_g, int N){
  
  //find the features uniqe to time series i
  uvec Uniq = find(m == 1 && F.col(i)==1);
  int n_Uniq = Uniq.n_elem;
  
  //propose birth if no unique features, otherwise pick a birth w/ prob 0.5
  double p_birth;
  if(n_Uniq==0){p_birth=1;}else{if(sum(F.col(i))==1){p_birth=1;}else{p_birth=0.5;}}
  
  //randomly select the window of data for birth-death proposal
  int L = randi(distr_param(L_min,L_max));
  int W_0 = randi(distr_param(0,T-L-1));
  int W_1 = W_0 + L;
  
  int k_remove;
  double Q_f;
  List out6(6);
  double Lik;
  int K_temp;
  vec f_i_prop(K_cur+1,fill::zeros);
  f_i_prop.head(K_cur) = F.col(i);
  mat n_k_i_prop(K_cur+1,N_groups,fill::zeros);
  n_k_i_prop.head_rows(K_cur) = n_k_i;
  
  if(randu()<p_birth){
    //propose a birth and account for likelihood of f_i and f_i_prop
    f_i_prop(K_cur) = 1;
    K_temp = K_cur+1;
    Q_f = log(0.5/(n_Uniq+1)) - log(p_birth);
    out6 = BirthGroup(Y_, Y_lags, F.col(i), Z, W_0, W_1, K_cur, T, group_i, D, r, 
                      K_mat, gamma, kappa, S_0, n_0, logdet_S_gb, logdet_S_bb, 
                      N_g, f_i_prop);
    Lik = Lik_base - log(n_Uniq+1);
    n_k_i_prop.col(i) = as<vec>(out6("n_k_i_new"));
  }else{  
    //propose a death of feature k_remove and account for likelihood of f_i and f_i_prop
    K_temp = K_cur;
    k_remove = Uniq(floor(n_Uniq*randu()));
    f_i_prop.shed_row(K_cur);
    f_i_prop(k_remove) = 0;
    Q_f = log(0.5) - log(0.5/n_Uniq);
    out6 = DeathGroup(Y_, Y_lags, F.col(i), Z, W_0, W_1, K_cur, T, group_i, D, r, 
                      K_mat, gamma, kappa, k_remove, S_0, n_0, logdet_S_gb, 
                      logdet_S_bb, N_g, f_i_prop);
    Lik = -Lik_base + log(n_Uniq);
    n_k_i_prop.shed_row(K_cur);
    n_k_i_prop.col(i) = as<vec>(out6("n_k_i_new"));
  }
  mat Z_prop = as<mat>(out6("Z_new"));
  mat TransMat_prop = as<mat>(out6("TransMat_new"));
  vec n_k_prop = sum(n_k_i_prop,1);
  double Q_z = as<double>(out6("Q_z"));
  vec logdet_S_gb_prop = as<vec>(out6("logdet_S_gb_prop"));
  vec logdet_S_bb_prop = as<vec>(out6("logdet_S_bb_prop"));
  
  //account for likelihood of Z_i and Z_i_prop
  Lik += Lik_Z_Z_prop(TransMat_i, TransMat_prop, gamma, kappa, F.col(i), f_i_prop, 
                      K_cur, K_temp, lambda);
  
  //account for likelihood of Y|Z
  for(int k=0; k<K_cur; k++){
    if(F(k,i)==1){
      Lik -= MargY(n_k(k), D, n_0, logdet_S_gb(k), logdet_S_bb(k), Marg_Prior);
    }
  }
  
  //account for likelihood of Y|Z_prop
  for(int k=0; k<K_temp; k++){
    if(f_i_prop(k) == 1){
      Lik += MargY(n_k_prop(k), D, n_0, logdet_S_gb_prop(k), logdet_S_bb_prop(k), Marg_Prior);
    }
  }
  
  //determine if you should accept the proposal or not
  if(log(randu()) < (Lik + (Q_z + Q_f)*anneal_temp)){
    mat F_prop(K_cur,N_groups, fill::zeros);
    int K_prop;
    if(K_temp == K_cur){
      K_prop = K_cur - 1;
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_remove){Z_prop(t,n) = Z_prop(t,n)-1;}
        }
      }
      F_prop = F;
      F_prop.shed_row(k_remove);
      n_k_prop.shed_row(k_remove);
      n_k_i_prop.shed_row(k_remove);
      logdet_S_gb_prop.shed_row(k_remove);
      logdet_S_bb_prop.shed_row(k_remove);
      TransMat_prop.shed_row(k_remove);
      TransMat_prop.shed_col(k_remove);
    }else{
      K_prop = K_cur + 1;
      F_prop.insert_rows(K_cur,1);
      F_prop.head_rows(K_cur) = F;
      F_prop(K_cur,i) = 1;
    }
    return List::create(
      _["accept"] = true,
      _["K"] = K_prop,
      _["k_remove"] = k_remove,
      _["logdet_S_gb"] = logdet_S_gb_prop,
      _["logdet_S_bb"] = logdet_S_bb_prop,
      _["Z"] = Z_prop,
      _["F"] = F_prop,
      _["n_k"] = n_k_prop,
      _["n_k_i"] = n_k_i_prop,
      _["TransMat"] = TransMat_prop
    );
  }else{
    return List::create(
      _["accept"] = false,
      _["K"] = K_cur,
      _["k_remove"] = K_cur,
      _["logdet_S_gb"] = logdet_S_gb,
      _["logdet_S_bb"] = logdet_S_bb,
      _["Z"] = Z,
      _["F"] = F,
      _["n_k"] = n_k,
      _["n_k_i"] = n_k_i,
      _["TransMat"] = TransMat_i
    );
  }
}


//Function to propose merge step for GBP-AR-HMM
// [[Rcpp::export]]
List MergeGroup(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z,
                int i, int j, int k_i, int k_j, int K_cur, int N, int T, int D, 
                int r, const mat& K_mat, const mat& S_0, int n_0, double gamma, 
                double kappa, const mat& target_Z, const cube& TransCube, 
                const field<uvec>& in_g, const vec& N_g, int N_groups){
  
  bool just_prob;
  if(target_Z(0,0)>=0){just_prob = true;}else{just_prob = false;}
  cube TransCube_prop(K_cur+1,K_cur+1,N_groups);
  TransCube_prop.subcube(0,0,0,K_cur-1,K_cur-1,N_groups-1) = TransCube;
  
  mat F_prop(K_cur+1, N_groups);
  F_prop.head_rows(K_cur) = F;
  F_prop.row(K_cur).fill(0);
  vec Has_i_or_j(N_groups, fill::zeros);
  for(int n=0; n<N_groups; n++){
    if(F(k_i,n)==1 || F(k_j,n)==1){
      F_prop(K_cur,n) = 1;
      Has_i_or_j(n) = 1;
    }
  }
  F_prop.row(k_i).fill(0);
  F_prop.row(k_j).fill(0);
  //determine the set of time series groups with feature i or j
  uvec ActiveSet = find(Has_i_or_j == 1);
  int n_active = ActiveSet.n_elem;
  //determine the set of features the time series in the active set possess
  vec TimesActive = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeat = find(TimesActive > 0);
  
  mat Z_prop = Z;
  int n_temp;
  for(int i_temp=0; i_temp<N_g(i); i_temp++){
    n_temp = in_g(i)(i_temp);
    for(int t=0; t<T; t++){
      if(Z_prop(t,n_temp)==k_i || Z_prop(t,n_temp)==k_j){Z_prop(t,n_temp) = K_cur;}
    }
  }
  for(int j_temp=0; j_temp<N_g(j); j_temp++){
    n_temp = in_g(j)(j_temp);
    for(int t=0; t<T; t++){
      if(Z_prop(t,n_temp)==k_i || Z_prop(t,n_temp)==k_j){Z_prop(t,n_temp) = K_cur;}
    }
  }
  
  cube A_new(D, D*r, K_cur+1);
  cube Sigma_new(D, D, K_cur+1);
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_prop), D, r, K_mat,
                                       S_0, n_0, ActiveFeat.n_elem, ActiveFeat, 
                                       empty, empty, false);
  A_new.slices(ActiveFeat) = as<cube>(outDHP("A"));
  Sigma_new.slices(ActiveFeat) = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur+1,K_cur+1);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  //propose new state sequences for all time series in active set (other than groups i and j)
  int g_temp;
  int n_k;
  int n_k_last;
  double q = 0;
  vec vecZ(T*N);
  for(int n=0; n<n_active; n++){
    g_temp = ActiveSet(n);
    if(g_temp != i && g_temp != j){
      n_k_last = 0;
      TransCube_prop.slice(g_temp).fill(0);
      uvec K_inc = find(F_prop.col(g_temp)==1);
      for(int subj=0; subj<N_g(g_temp); subj++){
        n_temp = in_g(g_temp)(subj);
        mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc), Sigma_new.slices(K_inc), 
                                      n_temp, F_prop.col(g_temp), K_cur+1, T, D);
        if(just_prob == false){
          List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T);
          q += as<double>(outZ("log_prob"));
          Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
          TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
          n_k_last += as<vec>(outZ("n_k_new"))(K_cur);
        }else{
          q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T, 
                             target_Z.col(n_temp));
          Z_prop.col(n_temp) = target_Z.col(n_temp);
          uvec inK_cur = find(target_Z.col(n_temp) == K_cur);
          n_k_last += inK_cur.n_elem;
        }
      }
      //update new state sufficient statistics if you added new timepoints
      if(n_k_last > 0){
        vecZ = vectorise(Z_prop);
        uvec inK = find(vecZ == K_cur);
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
    }
  }
  
  //propose state sequence for time series group i
  g_temp = i;
  TransCube_prop.slice(g_temp).fill(0);
  uvec K_inc_i = find(F_prop.col(g_temp)==1);
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_i), Sigma_new.slices(K_inc_i), 
                                  n_temp, F_prop.col(g_temp), K_cur+1, T, D);
    if(just_prob == false){
      List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T);
      q += as<double>(outZ("log_prob"));
      Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
      TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
    }else{
      q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T, 
                         target_Z.col(n_temp));
      Z_prop.col(n_temp) = target_Z.col(n_temp);
    }
  }
  
  //propose state sequence for time series group j
  g_temp = j;
  TransCube_prop.slice(g_temp).fill(0);
  uvec K_inc_j = find(F_prop.col(g_temp)==1);
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    mat log_Likeli = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_j), Sigma_new.slices(K_inc_j), 
                                  n_temp, F_prop.col(g_temp), K_cur+1, T, D);
    if(just_prob == false){
      List outZ = Sample_Z(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T);
      q += as<double>(outZ("log_prob"));
      Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
      TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
    }else{
      q += Sample_Z_prob(log_Likeli, eta_null, K_cur+1, F_prop.col(g_temp), T, 
                         target_Z.col(n_temp));
      Z_prop.col(n_temp) = target_Z.col(n_temp);
    }
  }
  
  if(just_prob == false){
    return List::create(
      _["Z_prop"] = Z_prop,
      _["F_prop"] = F_prop,
      _["q"] = q,
      _["TransCube"] = TransCube_prop,
      _["ActiveSet"] = ActiveSet
    );
  }else{
    return List::create(
      _["Z_prop"] = Z,
      _["F_prop"] = F,
      _["q"] = q,
      _["TransCube"] = TransCube,
      _["ActiveSet"] = ActiveSet
    );
  }
}


//Function to propose split step for GBP-AR-HMM
// [[Rcpp::export]]
List SplitGroup(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z, int i, 
                int j, int k_i, int K_cur, int N, int T, int D, int r, 
                const mat& K_mat, const mat& S_0, int n_0, double gamma, double kappa,
                double c, const mat& target_Z, const mat& target_F, const cube& TransCube, 
                const field<uvec>& in_g, const vec& N_g, int N_groups){
  
  bool just_prob;
  if(target_Z(0,0)>=0){just_prob = true;}else{just_prob = false;}
  cube TransCube_prop(K_cur+2,K_cur+2,N_groups);
  TransCube_prop.subcube(0,0,0,K_cur-1,K_cur-1,N_groups-1) = TransCube;
  
  mat F_prop(K_cur+2, N_groups, fill::zeros);
  F_prop.head_rows(K_cur) = F;
  F_prop.row(k_i).fill(0);
  F_prop(K_cur,i) = 1;
  F_prop(K_cur+1,j) = 1;
  //determine the set of time series groups with feature k_i
  uvec ActiveSet = find(F.row(k_i).t() == 1);
  int n_active = ActiveSet.n_elem;
  //determine the set of features the time series in the active set possess
  vec TimesActive = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeat = find(TimesActive > 0);
  
  mat Z_prop = Z;
  int n_temp;
  for(int i_temp=0; i_temp<N_g(i); i_temp++){
    n_temp = in_g(i)(i_temp);
    for(int t=0; t<T; t++){
      if(Z_prop(t,n_temp)==k_i){Z_prop(t,n_temp) = K_cur;}
    }
  }
  for(int j_temp=0; j_temp<N_g(j); j_temp++){
    n_temp = in_g(j)(j_temp);
    for(int t=0; t<T; t++){
      if(Z_prop(t,n_temp)==k_i){Z_prop(t,n_temp) = K_cur+1;}
    }
  }
  
  cube A_new(D, D*r, K_cur+2);
  cube Sigma_new(D, D, K_cur+2);
  vec empty(1,fill::zeros);
  List outDHP = DeterministicHMMParams(Y_, Y_lags, vectorise(Z_prop), D, r, K_mat, S_0, 
                                       n_0, ActiveFeat.n_elem, ActiveFeat, empty, empty, false);
  A_new.slices(ActiveFeat) = as<cube>(outDHP("A"));
  Sigma_new.slices(ActiveFeat) = as<cube>(outDHP("Sigma"));
  mat eta_null(K_cur+2,K_cur+2);
  eta_null.fill(gamma);
  eta_null.diag().fill(gamma+kappa);
  
  mat Y_hat(D, T);
  int m_a = 1;
  int m_b = 1;
  int scanned = 2;
  double q_z = 0;
  double q_f = 0;
  //p_ first element is p_10, second is p_01, third is p_11
  vec log_p(3);
  vec f_probs(3);
  vec f_star(K_cur+2);
  mat pi_n(K_cur+2, K_cur+2);
  vec n_k_temp(K_cur+2);
  int n_k_a;
  int n_k_b;
  int n_k;
  int f_pick;
  double L;
  int g_temp;
  vec temp3(3,fill::ones);
  vec vecZ(N*T);
  vec F_prop_temp(K_cur+2, fill::ones);
  //sample features allocations and state sequences from proposed split
  //or calculate log_prob of splitting to target F and Z
  for(int n=0; n<n_active; n++){
    g_temp = ActiveSet(n);
    cube log_Likeli(K_cur+2,T,N_g(g_temp));
    if(g_temp != i && g_temp != j){
      TransCube_prop.slice(g_temp).fill(0);
      F_prop_temp.head(K_cur) = F_prop.col(g_temp).head(K_cur);
      uvec K_inc = find(F_prop_temp==1);
      //get likelihood of each available state at each time point for given time series group
      for(int subj=0; subj<N_g(g_temp); subj++){
        n_temp = in_g(g_temp)(subj);
        log_Likeli.slice(subj) = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc), Sigma_new.slices(K_inc), 
                                              n_temp, F_prop_temp, K_cur+2, T, D);
      }
      
      //calculate log(p_10)
      f_star = F_prop.col(g_temp);
      f_star(K_cur) = 1;
      L = 0;
      for(int subj=0; subj<N_g(g_temp); subj++){
        L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli.slice(subj));
      }
      log_p(0) = log(m_a/(scanned+c)*(scanned-m_b+c)/(scanned+c)) + L;
      //calculate log(p_11)
      f_star(K_cur+1) = 1;
      L = 0;
      for(int subj=0; subj<N_g(g_temp); subj++){
        L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli.slice(subj));
      }
      log_p(2) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
      //calculate log(p_01)
      f_star(K_cur) = 0;
      L = 0;
      for(int subj=0; subj<N_g(g_temp); subj++){
        L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli.slice(subj));
      }
      log_p(1) = log(m_b/(scanned+c)*(scanned-m_a+c)/(scanned+c)) + L;
      f_probs = Multinom_prob_stable(log_p, 3, temp3);
      if(just_prob == false){
        //if not target allocation then sample feature allocations and get log_probs
        f_pick = Sample_Multinom(f_probs);
        if(f_pick == 0){
          F_prop(K_cur,g_temp) = 1;
          m_a++;
          q_f += log(f_probs(0));
        }
        if(f_pick == 1){
          F_prop(K_cur+1,g_temp) = 1;
          m_b++;
          q_f += log(f_probs(1));
        }
        if(f_pick == 2){
          F_prop(K_cur,g_temp) = 1;
          F_prop(K_cur+1,g_temp) = 1;
          m_a++;
          m_b++;
          q_f += log(f_probs(2));
        }
      }else{
        //or get log probability of target feature allocations
        m_a += target_F(K_cur,g_temp);
        m_b += target_F(K_cur+1,g_temp);
        if(target_F(K_cur,g_temp)==1 && target_F(K_cur+1,g_temp)==1){
          q_f += log(f_probs(2));
        }else{
          if(target_F(K_cur,g_temp)==1){
            q_f += log(f_probs(0));
          }
          if(target_F(K_cur+1,g_temp)==1){
            q_f += log(f_probs(1));
          }
        }
        F_prop.col(g_temp) = target_F.col(g_temp);
      }
      scanned++;
      n_k_a = 0;
      n_k_b = 0;
      
      //sample new state sequences given new feature allocations
      for(int subj=0; subj<N_g(g_temp); subj++){
        n_temp = in_g(g_temp)(subj);
        if(just_prob == false){
          List outZ = Sample_Z(log_Likeli.slice(subj), eta_null, K_cur+2, 
                               F_prop.col(g_temp), T);
          q_z += as<double>(outZ("log_prob"));
          Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
          TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
          n_k_temp = as<vec>(outZ("n_k_new"));
          n_k_a += n_k_temp(K_cur);
          n_k_b += n_k_temp(K_cur+1);
        }else{
          q_z += Sample_Z_prob(log_Likeli.slice(subj), eta_null, K_cur+2, 
                               F_prop.col(g_temp), T, target_Z.col(n_temp));
          Z_prop.col(n_temp) = target_Z.col(n_temp);
          uvec inK_cur = find(target_Z.col(n_temp) == K_cur);
          n_k_a += inK_cur.n_elem;
          uvec inK_cur1 = find(target_Z.col(n_temp) == K_cur+1);
          n_k_b += inK_cur1.n_elem;
        }
      }
      vecZ = vectorise(Z_prop);
      //update new state sufficient statistics if you added new timepoints
      if(n_k_a > 0){
        uvec inK = find(vecZ == K_cur);
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
      if(n_k_b > 0){
        uvec inK = find(vecZ == (K_cur+1));
        n_k = inK.n_elem;
        if(n_k<2){n_k=2;}
        List outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
        Sigma_new.slice(K_cur+1) = (as<mat>(outS("S_gb"))+S_0)/(n_k + n_0 - D - 1);
        A_new.slice(K_cur+1) = as<mat>(outS("S_b")) * as<mat>(outS("S_bb_inv"));
      }
    }
  }
  
  log_p.shed_row(0);
  f_probs.shed_row(0);
  int N_ij;
  if(N_g(i) >= N_g(j)){N_ij = N_g(i);}else{N_ij = N_g(j);}
  cube log_Likeli_ij(K_cur+2,T,N_ij);
  //sample new feature vector and state sequence for time series i
  g_temp = i;
  TransCube_prop.slice(g_temp).fill(0);
  F_prop_temp.head(K_cur) = F_prop.col(g_temp).head(K_cur);
  uvec K_inc_i = find(F_prop_temp==1);
  //get likelihood of each available state at each time point for time series group i
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    log_Likeli_ij.slice(subj) = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_i), Sigma_new.slices(K_inc_i), 
                                             n_temp, F_prop_temp, K_cur+2, T, D);
  }
  
  //calculate log(p_10)
  f_star = F_prop.col(g_temp);
  L = 0;
  for(int subj=0; subj<N_g(g_temp); subj++){
    L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli_ij.slice(subj));
  }
  log_p(0) = log(m_a/(scanned+c)*(scanned-m_b+c)/(scanned+c)) + L;
  //calculate log(p_11)
  f_star(K_cur+1) = 1;
  L = 0;
  for(int subj=0; subj<N_g(g_temp); subj++){
    L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli_ij.slice(subj));
  }
  log_p(1) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
  f_probs = Multinom_prob_stable(log_p, 2, temp3.head(2));
  if(just_prob == false){
    //if not target allocation then sample feature allocations and get log_probs
    f_pick = Sample_Multinom(f_probs);
    if(f_pick == 0){
      q_f += log(f_probs(0));
    }
    if(f_pick == 1){
      F_prop(K_cur+1,g_temp) = 1;
      q_f += log(f_probs(1));
    }
  }else{
    //or get log probability of target feature allocations
    if(target_F(K_cur+1,g_temp)==1){
      q_f += log(f_probs(1));
    }else{
      q_f += log(f_probs(0));
    }
    F_prop.col(g_temp) = target_F.col(g_temp);
  }  
  
  //sample new state sequences given new feature allocations
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    if(just_prob == false){
      List outZ = Sample_Z(log_Likeli_ij.slice(subj), eta_null, K_cur+2, 
                           F_prop.col(g_temp), T);
      q_z += as<double>(outZ("log_prob"));
      Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
      TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
    }else{
      q_z += Sample_Z_prob(log_Likeli_ij.slice(subj), eta_null, K_cur+2, 
                           F_prop.col(g_temp), T, target_Z.col(n_temp));
      Z_prop.col(n_temp) = target_Z.col(n_temp);
    }
  }
    
  //sample new feature vector and state sequence for time series group j
  g_temp = j;
  TransCube_prop.slice(g_temp).fill(0);  
  F_prop_temp.head(K_cur) = F_prop.col(g_temp).head(K_cur);
  uvec K_inc_j = find(F_prop_temp==1);
  //get likelihood of each available state at each time point for time series group j
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    log_Likeli_ij.slice(subj) = getLikeliMat(Y_, Y_lags, A_new.slices(K_inc_j), Sigma_new.slices(K_inc_j), 
                                             n_temp, F_prop_temp, K_cur+2, T, D);
  }
  
  //calculate log(p_01)
  f_star = F_prop.col(g_temp);
  L = 0;
  for(int subj=0; subj<N_g(g_temp); subj++){
    L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli_ij.slice(subj));
  }
  log_p(0) = log(m_b/(scanned+c)*(scanned-m_a+c)/(scanned+c)) + L;
  //calculate log(p_11)
  f_star(K_cur) = 1;
  L = 0;
  for(int subj=0; subj<N_g(g_temp); subj++){
    L += get_L_Y_given_f_theta(f_star, K_cur+2, T, eta_null, log_Likeli_ij.slice(subj));
  }
  log_p(1) = log(m_a/(scanned+c)*m_b/(scanned+c)) + L;
  f_probs = Multinom_prob_stable(log_p, 2, temp3.head(2));
  if(just_prob == false){
    //if not target allocation then sample feature allocations and get log_probs
    f_pick = Sample_Multinom(f_probs);
    if(f_pick == 0){
      q_f += log(f_probs(0));
    }
    if(f_pick == 1){
      F_prop(K_cur,g_temp) = 1;
      q_f += log(f_probs(1));
    }
  }else{
    //or get log probability of target feature allocations
    if(target_F(K_cur,g_temp)==1){
      q_f += log(f_probs(1));
    }else{
      q_f += log(f_probs(0));
    }
    F_prop.col(g_temp) = target_F.col(g_temp);
  }
  
  //sample new state sequences given new feature allocations
  for(int subj=0; subj<N_g(g_temp); subj++){
    n_temp = in_g(g_temp)(subj);
    if(just_prob == false){
      List outZ = Sample_Z(log_Likeli_ij.slice(subj), eta_null, K_cur+2, 
                           F_prop.col(g_temp), T);
      q_z += as<double>(outZ("log_prob"));
      Z_prop.col(n_temp) = as<vec>(outZ("Z_new"));
      TransCube_prop.slice(g_temp) += as<mat>(outZ("TransMat"));
    }else{
      q_z += Sample_Z_prob(log_Likeli_ij.slice(subj), eta_null, K_cur+2, 
                           F_prop.col(g_temp), T, target_Z.col(n_temp));
      Z_prop.col(n_temp) = target_Z.col(n_temp);
    }
  }
  
  if(just_prob == false){
    return List::create(
      _["Z_prop"] = Z_prop,
      _["F_prop"] = F_prop,
      _["q"] = q_z + q_f,
      _["TransCube"] = TransCube_prop,
      _["ActiveSet"] = ActiveSet
    );
  }else{
    return List::create(
      _["Z_prop"] = Z,
      _["F_prop"] = F,
      _["q"] = q_z + q_f,
      _["TransCube"] = TransCube,
      _["ActiveSet"] = ActiveSet
    );
  }
}


//Function to sample from BP-AR-HMM split-merge process for GBP-AR-HMM
// [[Rcpp::export]]
List SplitMergeGroup(const mat& Y_, const mat& Y_lags, const mat& F, const mat& Z,
                     double kappa, double gamma, double alpha, double c, int N, int T,
                     int K_cur, const vec& n_k, int D, int r, const mat& S_0, int n_0, 
                     double Marg_Prior, const vec& logdet_S_gb, const vec& logdet_S_bb, 
                     const mat& K_mat, const cube& TransCube, const mat& lambda,
                     double anneal_temp, int N_groups, const field<uvec>& in_g, 
                     const vec& N_g){
  
  //randomly select time series to select features from
  int i = floor(randu()*N_groups);
  int j = i;
  while(j == i){j = floor(randu()*N_groups);}
  
  //select features for split or merge
  List out3(3);
  out3 = FeatureSelect(Y_, Y_lags, F, Z, i, j, K_cur, n_k, D, r, n_0, 
                       Marg_Prior, logdet_S_gb, logdet_S_bb, K_mat);
  int k_i = as<int>(out3("k_i"));
  int k_j = as<int>(out3("k_j"));
  double q_k_fwd = as<double>(out3("q_k"));
  
  mat F_prop(K_cur+2, N_groups);
  mat Z_prop(T, N);
  cube TransCube_prop(K_cur+2, K_cur+2, N_groups);
  vec vecZ(T*N);
  vec n_k_prop(K_cur+2);
  vec logdet_S_gb_prop(K_cur+2);
  vec logdet_S_bb_prop(K_cur+2);
  double q_fwd;
  double q_rev;
  double q_k_rev;
  double sign;
  mat no_targ(1,1);
  no_targ.fill(-1); 
  mat target_Z = Z;
  uvec ActiveSet;
  List out5(5);
  List out4(4);
  bool merge_ind;
  if(k_i == k_j){
    //if k_i = k_j attempt a split move
    merge_ind = false;
    out5 = SplitGroup(Y_, Y_lags, F, Z, i, j, k_i, K_cur, N, T, D, r, 
                      K_mat, S_0, n_0, gamma, kappa, c, no_targ, no_targ,
                      TransCube, in_g, N_g, N_groups);
    F_prop = as<mat>(out5("F_prop"));
    Z_prop = as<mat>(out5("Z_prop"));
    TransCube_prop = as<cube>(out5("TransCube"));
    q_fwd = as<double>(out5("q"));
    ActiveSet = as<uvec>(out5("ActiveSet"));
    vecZ = vectorise(Z_prop);
    for(int k=0; k<(K_cur+2); k++){
      uvec inK = find(vecZ==k);
      n_k_prop(k) = inK.n_elem;
      out4 = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(out4("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(out4("S_bb")));
    }
    out3 = FeatureSelect(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur+2, n_k_prop,
                         D, r, n_0, Marg_Prior, logdet_S_gb_prop, 
                         logdet_S_bb_prop, K_mat, K_cur, K_cur+1);
    q_k_rev = as<double>(out3("q_k"));
    for(int n=0; n<N; n++){
      for(int t=0; t<T; t++){
        if(target_Z(t,n) == k_i){target_Z(t,n) = K_cur+2;}
      }
    }
    out5 = MergeGroup(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur, K_cur+1, K_cur+2, 
                      N, T, D, r, K_mat, S_0, n_0, gamma, kappa, target_Z, 
                      TransCube_prop, in_g, N_g, N_groups);
    q_rev = as<double>(out5("q"));
  }else{
    //otherwise attempt a merge move
    merge_ind = true;
    n_k_prop.shed_row(0);
    logdet_S_gb_prop.shed_row(0);
    logdet_S_bb_prop.shed_row(0);
    F_prop.shed_col(0);
    F_prop.shed_row(0);
    TransCube_prop.shed_col(0);
    TransCube_prop.shed_row(0);
    out5 = MergeGroup(Y_, Y_lags, F, Z, i, j, k_i, k_j, K_cur, N, T, D, r, 
                      K_mat, S_0, n_0, gamma, kappa, no_targ, TransCube, 
                      in_g, N_g, N_groups);
    F_prop = as<mat>(out5("F_prop"));
    Z_prop = as<mat>(out5("Z_prop"));
    TransCube_prop = as<cube>(out5("TransCube"));
    q_fwd = as<double>(out5("q"));
    ActiveSet = as<uvec>(out5("ActiveSet"));
    vecZ = vectorise(Z_prop);
    for(int k=0; k<(K_cur+1); k++){
      uvec inK = find(vecZ==k);
      n_k_prop(k) = inK.n_elem;
      out4 = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb_prop(k),sign,as<mat>(out4("S_gb")));
      log_det(logdet_S_bb_prop(k),sign,as<mat>(out4("S_bb")));
    }
    out3 = FeatureSelect(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur+1, n_k_prop,
                         D, r, n_0, Marg_Prior, logdet_S_gb_prop, 
                         logdet_S_bb_prop, K_mat, K_cur, K_cur);
    q_k_rev = as<double>(out3("q_k"));   
    for(int n=0; n<N; n++){
      for(int t=0; t<T; t++){
        if(target_Z(t,n) == k_i){target_Z(t,n) = K_cur+1;}
        if(target_Z(t,n) == k_j){target_Z(t,n) = K_cur+2;}
      }
    }
    mat target_F(K_cur+3,N_groups,fill::zeros);
    target_F.head_rows(K_cur) = F;
    target_F.row(k_i).fill(0);
    target_F.row(k_j).fill(0);
    target_F.row(K_cur+1) = F.row(k_i);
    target_F.row(K_cur+2) = F.row(k_j);
    out5 = SplitGroup(Y_, Y_lags, F_prop, Z_prop, i, j, K_cur, K_cur+1, N, T, D, r, 
                      K_mat, S_0, n_0, gamma, kappa, c, target_Z, target_F, 
                      TransCube_prop, in_g, N_g, N_groups);
    q_rev = as<double>(out5("q"));
  }
  
  vec m_Prop = sum(F_prop.cols(ActiveSet),1);
  uvec ActiveFeatProp = find(m_Prop > 0);
  int n_Feat_active_prop = ActiveFeatProp.n_elem;
  vec m_cur = sum(F.cols(ActiveSet),1);
  uvec ActiveFeat = find(m_cur > 0);
  int n_Feat_active = ActiveFeat.n_elem;
  int n_active = ActiveSet.n_elem;
  double Lik;
  
  //Account for likelihood of F and F_prop
  if(merge_ind == false){
    Lik = log(alpha) + log(c) - lgamma(N_groups+c) + 
      lgamma(m_Prop(K_cur)) + lgamma(N_groups-m_Prop(K_cur)+c) +
      lgamma(m_Prop(K_cur+1)) + lgamma(N_groups-m_Prop(K_cur+1)+c) -
      lgamma(m_cur(k_i)) - lgamma(N_groups-m_cur(k_i)+c);
  }else{
    Lik = -log(alpha) - log(c) + lgamma(N_groups+c) + 
      lgamma(m_cur(k_i)) + lgamma(N_groups-m_cur(k_i)+c) +
      lgamma(m_cur(k_j)) + lgamma(N_groups-m_cur(k_j)+c) -
      lgamma(m_Prop(K_cur)) - lgamma(N_groups-m_Prop(K_cur)+c);
  }
  
  int n_hist;
  vec check(n_Feat_active, fill::zeros);
  for(int k=0; k<(n_Feat_active-1); k++){
    if(check(k)==0){
      n_hist = 1;
      for(int k2=(k+1); k<n_Feat_active; k++){
        if(check(k2)==0 && sum(abs(F.row(ActiveFeat(k2)) - F.row(ActiveFeat(k)))) == 0){
          n_hist++;
          check(k2) = 1;
        }
      }
      Lik += log(factorial(n_hist));
    }
  }
  
  vec check_prop(n_Feat_active_prop, fill::zeros);
  for(int k=0; k<(n_Feat_active_prop-1); k++){
    if(check_prop(k)==0){
      n_hist = 1;
      for(int k2=(k+1); k<n_Feat_active_prop; k++){
        if(check_prop(k2)==0 && sum(abs(F_prop.row(ActiveFeatProp(k2)) - F_prop.row(ActiveFeatProp(k)))) == 0){
          n_hist++;
          check_prop(k2) = 1;
        }
      }
      Lik -= log(factorial(n_hist));
    }
  }
  
  //account for likelihood of Z and Z_prop
  int g_temp;
  int K_temp = F_prop.n_rows;
  for(int n=0; n<n_active; n++){
    g_temp = ActiveSet(n);
    Lik += Lik_Z_Z_prop(TransCube.slice(g_temp), TransCube_prop.slice(g_temp), 
                        gamma, kappa, F.col(g_temp), F_prop.col(g_temp), 
                        K_cur, K_temp, lambda);
  }
  
  //account for likelihood of Y|Z and Y_Z_prop
  int k_temp1;
  int k_temp2;
  int int_temp;
  if(merge_ind==true){
    int_temp = n_Feat_active_prop;
    k_temp2 = ActiveFeat(n_Feat_active-1);
    Lik -= MargY(n_k(k_temp2), D, n_0, logdet_S_gb(k_temp2), 
                 logdet_S_bb(k_temp2), Marg_Prior);
  }else{
    int_temp = n_Feat_active;
    k_temp1 = ActiveFeatProp(n_Feat_active_prop-1);
    Lik += MargY(n_k_prop(k_temp1), D, n_0, logdet_S_gb_prop(k_temp1), 
                 logdet_S_bb_prop(k_temp1), Marg_Prior);    
  }
  for(int k=0; k<int_temp ; k++){
    k_temp1 = ActiveFeatProp(k);
    k_temp2 = ActiveFeat(k);
    Lik += MargY(n_k_prop(k_temp1), D, n_0, logdet_S_gb_prop(k_temp1), 
                 logdet_S_bb_prop(k_temp1), Marg_Prior) - 
           MargY(n_k(k_temp2), D, n_0, logdet_S_gb(k_temp2), 
                 logdet_S_bb(k_temp2), Marg_Prior);
  }
  
  //determine if you should accept the proposal or not
  if(log(randu()) < (Lik + (q_rev - q_fwd + q_k_rev - q_k_fwd)*anneal_temp)){
    if(merge_ind==false){
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_i){Z_prop(t,n) = Z_prop(t,n)-1;}
        }
      }
      F_prop.shed_row(k_i);
      n_k_prop.shed_row(k_i);
      logdet_S_gb_prop.shed_row(k_i);
      logdet_S_bb_prop.shed_row(k_i);
      TransCube_prop.shed_row(k_i);
      TransCube_prop.shed_col(k_i);
    }else{
      for(int n=0; n<N; n++){
        for(int t=0; t<T; t++){
          if(Z_prop(t,n)>k_i && Z_prop(t,n)>k_j){
            Z_prop(t,n) = Z_prop(t,n)-2;
          }else{
            if(Z_prop(t,n)>k_i || Z_prop(t,n)>k_j){
              Z_prop(t,n) = Z_prop(t,n)-1;
            }
          }
        }
      }
      if(k_i > k_j){
        F_prop.shed_row(k_i);
        F_prop.shed_row(k_j);
        n_k_prop.shed_row(k_i);
        n_k_prop.shed_row(k_j);
        logdet_S_gb_prop.shed_row(k_i);
        logdet_S_gb_prop.shed_row(k_j);
        logdet_S_bb_prop.shed_row(k_i);
        logdet_S_bb_prop.shed_row(k_j);
        TransCube_prop.shed_row(k_i);
        TransCube_prop.shed_col(k_i);
        TransCube_prop.shed_row(k_j);
        TransCube_prop.shed_col(k_j);
      }else{
        F_prop.shed_row(k_j);
        F_prop.shed_row(k_i);
        n_k_prop.shed_row(k_j);
        n_k_prop.shed_row(k_i);
        logdet_S_gb_prop.shed_row(k_j);
        logdet_S_gb_prop.shed_row(k_i);
        logdet_S_bb_prop.shed_row(k_j);
        logdet_S_bb_prop.shed_row(k_i);
        TransCube_prop.shed_row(k_j);
        TransCube_prop.shed_col(k_j);
        TransCube_prop.shed_row(k_i);
        TransCube_prop.shed_col(k_i);
      }
    }
    
    return List::create(
      _["accept"] = true,
      _["merge_ind"] = merge_ind,
      _["logdet_S_gb"] = logdet_S_gb_prop,
      _["logdet_S_bb"] = logdet_S_bb_prop,
      _["Z"] = Z_prop,
      _["F"] = F_prop,
      _["n_k"] = n_k_prop,
      _["TransCube"] = TransCube_prop
    );
  }else{
    return List::create(
      _["accept"] = false,
      _["merge_ind"] = merge_ind,
      _["logdet_S_gb"] = logdet_S_gb,
      _["logdet_S_bb"] = logdet_S_bb,
      _["Z"] = Z,
      _["F"] = F,
      _["n_k"] = n_k,
      _["TransCube"] = TransCube
    );
  }
}




//Function to fit Sticky HDP VAR HMM model with MNIW conjugate prior
// [[Rcpp::export]]
List GBP_AR_HMM(const cube& Y, vec Groups, int N_samples, int L_min, int L_max,
                int r=1, double K_scale=1, int n_0=0, int K_init=1, int PrintEvery=100,
                int KeepEvery=1, int BDper=1, int SMper=1, double burnin=5000,
                int ProcessPer=1, bool reduce=true, int SMwarm=0,
                double c=1, double a_c=1, double b_c=1, double sigma_c_sq=1,
                double alpha=1, double a_alpha=1, double b_alpha=1,
                double gamma=1, double a_gamma=1, double b_gamma=1, double sigma_gamma_sq=1,
                double kappa=1, double a_kappa=1, double b_kappa=0.01, double sigma_kappa_sq=50){

  double T = Y.n_cols - r;
  int D = Y.n_rows;
  int N = Y.n_slices;

  //reshape Y into a DxT*N dimension matrix by concatenating slices and dropping first r time points of each series
  //and create corresponding design matrix of lagged values
  mat Y_(D,T*N);
  mat Y_lags(D*r,T*N);
  for(int n=0; n<N; n++){
    Groups(n)--;
    Y_.cols(n*T,(n+1)*T-1) = Y.slice(n).cols(r,T+r-1);
    for(int t=0; t<T; t++){
      Y_lags.submat(0, n*T+t, D*r-1, n*T+t) = vectorise(fliplr(Y.slice(n).cols(t, t+r-1)));
    }
  }

  //set Inverse Wishart prior hyperparameters to those suggested by Fox
  vec Y_bar(D);
  mat S_0(D,D,fill::zeros);
  for(int n=0; n<N; n++){
    Y_bar = mean(Y.slice(n).cols(1,T+r-1)-Y.slice(n).cols(0,T+r-2),1);
    for(int t=1; t<(T+r); t++){
      S_0 += (Y.slice(n).col(t)-Y.slice(n).col(t-1)-Y_bar)*(Y.slice(n).col(t)-Y.slice(n).col(t-1)-Y_bar).t();
    }
  }
  S_0 = 0.75*S_0/(T+r)/N;
  if(n_0 == 0){n_0 = D*r + 2;}
  mat K_mat = K_scale*eye(D*r, D*r);
  //cache prior parts of marginal Y likelihood for later since it remains constant
  double det1;
  double det2;
  double sign;
  log_det(det1,sign,S_0);
  log_det(det2,sign,K_mat);
  double Marg_Prior = -logMultiGamma(n_0/2,D) + (n_0/2)*det1 + det2/2;
  
  int K_cur = K_init;
  vec K_hist(N_samples);
  mat lambda(K_cur+3,K_cur+3);
  lambda.fill(gamma);
  lambda.diag().fill(gamma+kappa);
  vec gamma_hist(N_samples);
  vec kappa_hist(N_samples);
  vec alpha_hist(N_samples);
  vec c_hist(N_samples);

  //initialize A and Sigma for first features to posterior draws from full population
  field<cube> A(N_samples);
  cube A_cur(D, D*r, K_cur);
  field<cube> Sigma(N_samples);
  cube Sigma_cur(D, D, K_cur);
  List outS = get_MNIW_SS(Y_, Y_lags, K_mat, D, r);
  for(int k=0; k<K_cur; k++){
    List out2 = Sample_MNIW(D, r, as<mat>(outS("S_gb")), as<mat>(outS("S_bb_inv")),
                            as<mat>(outS("S_b")), S_0, n_0, floor(N*T/K_cur));
    A_cur.slice(k) = as<mat>(out2("A_new"));
    Sigma_cur.slice(k) = as<mat>(out2("Sigma_new"));
  }
  
  //inialize so that all time series have all initial features
  int N_groups = max(Groups)+1;
  mat F(K_cur, N_groups, fill::ones);
  vec m = sum(F,1);
  field<uvec> in_g(N_groups);
  vec N_g(N_groups);
  for(int g=0; g<N_groups; g++){
    in_g(g) = find(Groups==g);
    N_g(g) = in_g(g).n_elem;
  }
  
  //calculate likelihood of data in each available feature for each time series
  cube log_Likeli(K_cur, T, N);  
  vec F_ones(K_cur, fill::ones);
  for(int g=0; g<N_groups; g++){
    for(int subj=0; subj<N_g(g); subj++){
      int n_temp = in_g(g)(subj);
      log_Likeli.slice(n_temp) = getLikeliMat(Y_, Y_lags, A_cur, Sigma_cur, n_temp, F_ones, K_cur, T, D);
    }
  }
  
  //initiate transition distributions, eta, for each subject
  cube eta(K_cur, K_cur, N_groups, fill::ones);

  //initiate state sequences and sufficient statistics
  cube Z(T, N, N_samples);
  mat Z_cur(T, N);
  mat n_k_i(K_cur, N_groups, fill::zeros);
  cube TransCube(K_cur, K_cur, N_groups, fill::zeros);
  vec logdet_S_gb(K_cur);
  vec logdet_S_bb(K_cur);
  
  //MCMC sampler with burn-in with annealing to improve mixing
  int n_sample=0;
  int n_thin=1;
  int burn_iter = 0;
  int warm_iter = 0;
  double anneal_temp = (burn_iter + pow(10,-323))/burnin;
  double Lik_base = log(alpha) + log(c) + lgamma(N_groups-1+c) - lgamma(N_groups+c);
  while(n_sample < N_samples){
    //cout<<"A";
    //sample shared feature assignments
    for(int n=0; n<N_groups; n++){
      F.col(n) = sample_F_group(F.col(n), K_cur, N_groups, T, D, m-F.col(n), c,
                                eta.slice(n), log_Likeli.slices(in_g(n)), N_g(n));
      m = sum(F,1);
    }
    //cout<<"B";

    //sample state sequences
    TransCube.zeros(K_cur, K_cur, N_groups);
    n_k_i.zeros(K_cur,N_groups);
    for(int g=0; g<N_groups; g++){
      for(int subj=0; subj<N_g(g); subj++){
        int n_temp = in_g(g)(subj);
        List outZ = Sample_Z(log_Likeli.slice(n_temp), eta.slice(g), K_cur, 
                             F.col(g), T);
        Z_cur.col(n_temp) = as<vec>(outZ("Z_new"));
        TransCube.slice(g) += as<mat>(outZ("TransMat"));
        n_k_i.col(g) += as<vec>(outZ("n_k_new"));
      }
    }
    vec n_k = sum(n_k_i,1);
    vec vecZ = vectorise(Z_cur);
    for(int k=0; k<K_cur; k++){
      uvec inK = find(vecZ == k);
      outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb(k),sign,as<mat>(outS("S_bb")));
    }
    //cout<<"C";

    //propose birth/death step for each time series group BDper times
    int K_old = K_cur;
    if(warm_iter >= SMwarm){
      for(int BD_iter=0; BD_iter<BDper; BD_iter++){
        for(int g=0; g<N_groups; g++){
          List outBD = BirthDeathGroup(Y_, Y_lags, F, Z_cur, m, L_min, L_max, T, K_cur, g,
                                       r, D, kappa, gamma, K_mat, N_groups, TransCube.slice(g), 
                                       Lik_base, logdet_S_gb, logdet_S_bb, n_k_i, n_k, n_0, 
                                       S_0, anneal_temp, lambda, Marg_Prior, in_g(g), N_g(g), N);
          if(as<bool>(outBD("accept")) == true){
            if(as<int>(outBD("K")) > K_cur){
              //cout<<"Birth";
              TransCube.insert_rows(K_cur,1);
              TransCube.insert_cols(K_cur,1);
              lambda.insert_rows(K_cur,1);
              lambda.insert_cols(K_cur,1);
              lambda.fill(gamma);
              lambda.diag().fill(gamma+kappa);
              K_cur++;
            }else{
              //cout<<"Death";
              int k_remove = as<int>(outBD("k_remove"));
              TransCube.shed_row(k_remove);
              TransCube.shed_col(k_remove);
              lambda.shed_row(k_remove);
              lambda.shed_col(k_remove);
              K_cur = K_cur - 1;
            }
            F.set_size(K_cur,N_groups);
            mat F_temp = as<mat>(outBD("F"));
            F = F_temp;
            m.set_size(K_cur);
            m = sum(F,1);
            logdet_S_gb.set_size(K_cur);
            vec logdet_S_gb_temp = as<vec>(outBD("logdet_S_gb"));
            logdet_S_gb = logdet_S_gb_temp;
            logdet_S_bb.set_size(K_cur);
            vec logdet_S_bb_temp = as<vec>(outBD("logdet_S_bb"));
            logdet_S_bb = logdet_S_bb_temp;
            n_k.set_size(K_cur);
            vec n_k_temp = as<vec>(outBD("n_k"));
            n_k = n_k_temp;
            n_k_i.set_size(K_cur, N_groups);
            mat n_k_i_temp = as<mat>(outBD("n_k_i"));
            n_k_i = n_k_i_temp;
            mat Z_temp = as<mat>(outBD("Z"));
            Z_cur = Z_temp;
            mat TransMat_temp = as<mat>(outBD("TransMat"));
            TransCube.slice(g) = TransMat_temp;
          }
        }
      }    
    }

    //cout<<"D";
    for(int SM_iter=0; SM_iter<SMper; SM_iter++){
      //propose SMper many split merge moves
      List outSM = SplitMergeGroup(Y_, Y_lags, F, Z_cur, kappa, gamma, alpha, c, N, T,
                                   K_cur, n_k, D, r, S_0, n_0, Marg_Prior, logdet_S_gb,
                                   logdet_S_bb, K_mat, TransCube, lambda, anneal_temp, 
                                   N_groups, in_g, N_g);
      if(as<bool>(outSM("accept")) == true){
        if(as<bool>(outSM("merge_ind")) == false){
          //cout<<"Split";
          lambda.insert_rows(K_cur,1);
          lambda.insert_cols(K_cur,1);
          lambda.fill(gamma);
          lambda.diag().fill(gamma+kappa);
          K_cur++;
        }else{
          //cout<<"Merge";
          lambda.shed_row(0);
          lambda.shed_col(0);
          K_cur = K_cur - 1;
        }
        F.set_size(K_cur,N_groups);
        mat F_temp = as<mat>(outSM("F"));
        F = F_temp;
        m.set_size(K_cur);
        m = sum(F,1);
        logdet_S_gb.set_size(K_cur);
        vec logdet_S_gb_temp = as<vec>(outSM("logdet_S_gb"));
        logdet_S_gb = logdet_S_gb_temp;
        logdet_S_bb.set_size(K_cur);
        vec logdet_S_bb_temp = as<vec>(outSM("logdet_S_bb"));
        logdet_S_bb = logdet_S_bb_temp;
        n_k.set_size(K_cur);
        vec n_k_temp = as<vec>(outSM("n_k"));
        n_k = n_k_temp;
        mat Z_temp = as<mat>(outSM("Z"));
        Z_cur = Z_temp;
        TransCube.set_size(K_cur, K_cur, N_groups);
        cube TransCube_temp = as<cube>(outSM("TransCube"));
        TransCube = TransCube_temp;
      }
    }
    //resize structures to match number of states currently considered
    if(K_old != K_cur){
      A_cur.set_size(D, D*r, K_cur);
      Sigma_cur.set_size(D, D, K_cur);
      eta.set_size(K_cur,K_cur,N_groups);
      log_Likeli.set_size(K_cur,T,N);
    }
    //cout<<"E";

    //sample state emission parameters and calculate new likelihoods
    vecZ = vectorise(Z_cur);
    for(int k=0; k<K_cur; k++){
      uvec inK = find(vecZ == k);
      outS = get_MNIW_SS(Y_.cols(inK), Y_lags.cols(inK), K_mat, D, r);
      log_det(logdet_S_gb(k),sign,as<mat>(outS("S_gb")));
      log_det(logdet_S_bb(k),sign,as<mat>(outS("S_bb")));
      List  out2 = Sample_MNIW(D, r, as<mat>(outS("S_gb")), as<mat>(outS("S_bb_inv")), 
                               as<mat>(outS("S_b")), S_0, n_0, n_k(k));
      A_cur.slice(k) = as<mat>(out2("A_new"));
      Sigma_cur.slice(k) = as<mat>(out2("Sigma_new"));
    }
    F_ones.ones(K_cur);
    for(int g=0; g<N_groups; g++){
      for(int subj=0; subj<N_g(g); subj++){
        int n_temp = in_g(g)(subj);
        log_Likeli.slice(n_temp) = getLikeliMat(Y_, Y_lags, A_cur, Sigma_cur, n_temp, F_ones, K_cur, T, D);
      }
    }
    //cout<<"F";

    //sample state transition parameters
    for(int n=0; n<N_groups; n++){
      eta.slice(n) = sample_eta(TransCube.slice(n), gamma, kappa, K_cur, sum(F.col(n)));
    }

    //sample hyperparameters, repeat many times to improve acceptance (or don't sample at all)
    for(int ProcIter=0; ProcIter<ProcessPer; ProcIter++){
      alpha = sample_alpha(a_alpha, b_alpha, c, K_cur, N_groups);
      c = sample_c(c, a_c, b_c, alpha, sigma_c_sq, K_cur, N_groups, m);
      gamma = sample_gamma(gamma, a_gamma, b_gamma, sigma_gamma_sq, N_groups,
                           K_cur, kappa, F, eta);
      kappa = sample_kappa(kappa, a_kappa, b_kappa, sigma_kappa_sq, N_groups,
                           K_cur, gamma, F, eta);
      lambda.fill(gamma);
      lambda.diag().fill(gamma+kappa);
      Lik_base = log(alpha) + log(c) + lgamma(N_groups-1+c) - lgamma(N_groups+c);
    }

    //Only save parameter values every KeepEvery iterations
    if(warm_iter < SMwarm){
      warm_iter++;
      if(warm_iter%PrintEvery == 0){cout << warm_iter << " Split-Merge warmup samples complete" << endl;}
    }else{
      if(burn_iter < burnin){
        burn_iter++;
        anneal_temp = (burn_iter+pow(10,-323))/burnin;
        if(burn_iter%PrintEvery == 0){cout << burn_iter << " burn-in samples complete" << endl;}
      }else{
        if(n_thin%KeepEvery==0){
          if(reduce == true){
            vec used0 = unique(Z_cur);
            uvec used = conv_to<uvec>::from(used0);
            cube A_out = A_cur.slices(used);
            cube Sigma_out = Sigma_cur.slices(used);
            mat Z_out(T,N);
            vec key(K_cur);
            key.fill(-1);
            for(int k=0; k<K_cur; k++){
              uvec temp_ind = find(used0 == k);
              if(temp_ind.n_elem>0){key(k) = temp_ind(0)+1;}
            }
            for(int n1=0; n1<T; n1++){
              for(int n2=0; n2<N; n2++){
                Z_out(n1,n2) = key(Z_cur(n1,n2));
              }
            }
            A(n_sample) = A_out;
            Sigma(n_sample) = Sigma_out;
            Z.slice(n_sample) = Z_out;
          }else{
            A(n_sample) = A_cur;
            Sigma(n_sample) = Sigma_cur;
            Z.slice(n_sample) = Z_cur;
          }
          K_hist(n_sample) = K_cur;
          gamma_hist(n_sample) = gamma;
          kappa_hist(n_sample) = kappa;
          alpha_hist(n_sample) = alpha;
          c_hist(n_sample) = c;
          n_sample++;
          n_thin=1;
          if(n_sample%PrintEvery == 0){cout << n_sample<< " posterior samples complete" << endl;}
        }else{
          n_thin++;
        }
      } 
    }
  }

  return List::create(
    _["Sigma"] = Sigma,
    _["A"] = A,
    _["Z"] = Z,
    _["K"] = K_hist,
    _["gamma"] = gamma_hist,
    _["kappa"] = kappa_hist,
    _["alpha"] = alpha_hist,
    _["c"] = c_hist
  );
}



// //Function to reduce size of output files by removing empty state info
// // [[Rcpp::export]]
// List Reduce2(List Fit){
//   cube A = as<field<cube>>(Fit("A"))(0);
//   cube Sigma = as<field<cube>>(Fit("Sigma"))(0);
//   mat Z = as<cube>(Fit("Z")).slice(0);
//   mat Z_out(Z.n_rows,Z.n_cols);
//   mat Ones(Z.n_rows,Z.n_cols,fill::ones);
//   Z = Z - Ones;
//   
//   vec used0 = unique(Z);
//   uvec used = conv_to<uvec>::from(used0);
//   cube A_out = A.slices(used);
//   cube Sigma_out = Sigma.slices(used);
//   int K = A.n_slices;
//   vec key(K);
//   key.fill(-1);
//   for(int k=0; k<K; k++){
//     uvec temp_ind = find(used0 == k);
//     if(temp_ind.n_elem>0){key(k) = temp_ind(0);}
//   }
//   for(int n1=0; n1<Z.n_rows; n1++){
//     for(int n2=0; n2<Z.n_cols; n2++){
//       Z_out(n1,n2) = key(Z(n1,n2));
//     }
//   }
//   return List::create(
//     _["Sigma_out"] = Sigma_out,
//     _["Sigma_in"] = Sigma,
//     _["A_out"] = A_out,
//     _["A_in"] = A,
//     _["Z_out"] = Z_out,
//     _["Z_in"] = Z,
//     _["used0"] = used0,
//     _["used"] = used
//   );
// }





//Function to calculated expected pairwise allocation matrix from posterior cluster samples
// [[Rcpp::export]]
mat get_EPAM(const mat& PostClust){
  int n = PostClust.n_cols;
  int n_samp = PostClust.n_rows;
  mat EPAM(n,n);
  vec diff(n_samp);
  double temp;
  for(int i=0; i<n; i++){
    EPAM(i,i) = 1;
    for(int j=(i+1); j<n; j++){
      diff = PostClust.col(i) - PostClust.col(j);
      uvec eq = find(diff == 0);
      temp = eq.n_elem;
      EPAM(i,j) = temp/n_samp;
      EPAM(j,i) = temp/n_samp;
    }
  }
  return EPAM;
}



