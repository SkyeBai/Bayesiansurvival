data {
  // dimensions
  int<lower=1> N;             // number of observations
  int<lower=1> M;             // number of predictors
  
  // observations
  matrix[N, M] x;             // predictors for observation n
  vector[N] y;                // time for observation n
  vector[N] event;            // event status (1:event, 0:censor) for obs n

  real<lower=0> scale_global; // scale for the half-t prior for tau
  real<lower=1> nu_global;	  // degrees of freedom for the half-t priors for tau
  real<lower=1> nu_local;		  // degrees of freedom for the half-t priors for lambdas
  real<lower=0> slab_scale;   
  real<lower=1> slab_df;      // degrees of freedom for the slab
}


transformed data {
  real<lower=0> tau_mu;
  real<lower=0> tau_al;
  tau_mu = 10.0;
  tau_al = 10.0;
}

parameters { 
  real <lower=0.0>alpha_raw;
  vector[M] z; 

  real <lower=0.0> tau;         // global shrinkage parameter
  vector <lower=0.0>[M] lambda; // local shrinkage parameter

  real<lower = 0.0> c2;
  real<lower = 0.0>  mu;
}

transformed parameters {
  vector[M] beta;
  real <lower=0.0> alpha;
  vector [N] lp;
  real<lower=0.0> c;
  vector<lower=0.0>[M] lambda_tilde;
  
  c = sqrt(c2)*slab_scale;
  lambda_tilde = sqrt(c^2 * square(lambda)) ./ (c^2 + tau^2 * square(lambda));
  
  beta = z .* lambda_tilde*tau;
  
  alpha = exp(tau_al * alpha_raw);
  for (n in 1:N) {
    lp[n] = mu + dot_product(x[n], beta);
  }
}

model {
  // priors
  lambda ~ student_t(nu_local, 0, 1.0);  //cauchy(0, 1); //
  z ~ normal(0, 1);
  c2 ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  tau ~ cauchy(0, scale_global^2);//cauchy(0, 1);// normal(0, scale_global^2);//
  alpha_raw ~ normal(0.0, 1.0);
  mu ~ normal(0.0, tau_mu);

  // likelihood
  for (n in 1:N) {
      if (event[n] == 1){
          y[n] ~ weibull(alpha, exp(-(lp[n])/alpha));

      }
      else
          target += (weibull_lccdf(y[n] | alpha, exp(-(lp[n])/alpha)));
  }
}
generated quantities {
  vector[N] yhat_uncens;
  vector[N] log_lik;
  
  for (n in 1:N) {
      yhat_uncens[n] = weibull_rng(alpha, exp(-(lp[n])/alpha));
      if (event[n]==1) {
          log_lik[n] = weibull_lpdf(y[n] | alpha, exp(-(lp[n])/alpha));
      } else {
          log_lik[n] = weibull_lccdf(y[n] | alpha, exp(-(lp[n])/alpha));
      }
  }
}
