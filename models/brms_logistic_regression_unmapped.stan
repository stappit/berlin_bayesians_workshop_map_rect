functions {
}
data {
  int<lower=1> N;  // number of observations
  int Y[N];  // response variable
  // data for group-level effects of ID 1
  int<lower=1> N_1;
  int<lower=1> M_1;
  int<lower=1> J_1[N];
  vector[N] Z_1_1;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
}
parameters {
  real temp_Intercept;  // temporary intercept
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  vector[N_1] z_1[M_1];  // unscaled group-level effects
}
transformed parameters {
  // group-level effects
  vector[N_1] r_1_1 = (sd_1[1] * (z_1[1]));
}
model {
  vector[N] mu = temp_Intercept + rep_vector(0, N);
  for (n in 1:N) {
    mu[n] += r_1_1[J_1[n]] * Z_1_1[n];
  }
  // priors including all constants
  target += normal_lpdf(temp_Intercept | 0, 1);
  target += normal_lpdf(sd_1 | 0, 1)
    - 1 * normal_lccdf(0 | 0, 1);
  target += normal_lpdf(z_1[1] | 0, 1);
  // likelihood including all constants
  if (!prior_only) {
    target += bernoulli_logit_lpmf(Y | mu);
  }
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = temp_Intercept;
}
