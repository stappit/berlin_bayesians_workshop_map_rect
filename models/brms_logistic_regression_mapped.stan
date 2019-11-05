functions {
  vector lp(vector global, vector local, real[] xr, int[] xi) {
    int M = xi[1];
    int Y[M] = xi[2:M+1];
    real mu = global[1];    // intercept
    real sigma = global[2]; // scale of group-level effect
    real alpha = local[1];  // unscaled group_level effect

    real ll = bernoulli_logit_lpmf(Y | rep_vector(mu + alpha * sigma, M));

    return [ll]';
  }

  int[] count(int[] factr, int L) {
    int N = size(factr);
    int counts[L] = rep_array(0, L);
    for (i in 1:N) {
      counts[factr[i]] += 1;
    }
    return counts;
  }
}

data {
  int<lower=1> N;  // number of observations
  int Y[N];  // response variable
  // data for group-level effects of ID 1
  int<lower=1> N_1; // number of levels
  int<lower=1> M_1;
  int<lower=1> J_1[N]; // the level of each observation
  vector[N] Z_1_1;
  int prior_only;  // should the likelihood be ignored?
}

transformed data {
  int<lower = 0, upper = N> counts[N_1] = count(J_1, N_1); // number of observations in each level

  int<lower = 1> S = max(counts) + 1; // size of each shard

  int xi[N_1, S];
  real xr[N_1, S];

  int<lower = 1> p[N_1] = rep_array(2, N_1); // position of next datapoint
  xi[, 1] = counts; // first entry is the number of datapoints
  for (i in 1:N) {
    int shard = J_1[i]; // shards are indexed by the levels
    xi[shard, p[shard]] = Y[i];
    p[shard] += 1;
  }
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
  vector[2] global = [temp_Intercept, sd_1[1]]';
  vector[1] local[N_1];
  for (j in 1:N_1) { // convert [{1, 2, 3, ...}] into [{1}, {2}, {3}, ...]
    local[j] = [z_1[1][j]]';
  }

  // priors including all constants
  target += normal_lpdf(temp_Intercept | 0, 1);
  target += normal_lpdf(sd_1 | 0, 1)
    - 1 * normal_lccdf(0 | 0, 1);
  target += normal_lpdf(z_1[1] | 0, 1);

  // likelihood including all constants
  if (!prior_only) {
    target += sum(
      map_rect(lp, global, local, xr, xi)
    );
  }
}

generated quantities {
  // actual population-level intercept
  real b_Intercept = temp_Intercept;
}
