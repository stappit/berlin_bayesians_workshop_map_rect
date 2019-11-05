data {
  int<lower=1> N;                     // number of observations
  int<lower=1> L;                     // number of levels
  int<lower = 0, upper = 1> y[N];     // response variable
  int<lower = 1, upper = L> level[N]; // level of each observation
}

parameters {
  real mu;               // intercept
  real<lower = 0> sigma; // group-level standard deviation
  vector[L] alpha;       // unscaled group-level effects
}

transformed parameters {
  vector[N] logit_p;
  for (i in 1:N) {
    logit_p[i] = mu + sigma * alpha[level[i]];
  }
}

model {
  mu ~ normal(0, 1);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  y ~ bernoulli_logit_lpmf(logit_p);
}
