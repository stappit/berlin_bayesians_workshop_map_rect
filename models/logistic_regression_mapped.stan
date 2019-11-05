functions {
  // This function gets applied to each shard.
  // Global parameters are sent to every node.
  // Local parameters are only sent to that node computing that shard
  vector lp(vector global, vector local, real[] xr, int[] xi) {
    int M = xi[1];
    int y[M] = xi[2:M+1];
    real mu = global[1];    // intercept
    real sigma = global[2]; // scale of group-level effect
    real alpha = local[1];  // unscaled group_level effect

    real ll = bernoulli_logit_lpmf(y | rep_vector(mu + alpha * sigma, M));

    return [ll]';
  }

  // Count the number of observations per level
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
  int<lower=1> N;        // number of observations
  int y[N];              // response variable
  int<lower=1> L;        // number of levels
  int<lower=1> level[N]; // the level of each observation
}

transformed data {
  int<lower = 0, upper = N> counts[L] = count(level, L); // number of observations in each level

  int<lower = 1> S = max(counts) + 1; // size of each shard

  // both must have the same size for the first dimension (L)
  int xi[L, S];
  real xr[L, S];

  int<lower = 1> pos[L] = rep_array(2, L); // position of next datapoint
  xi[, 1] = counts; // first entry is the number of datapoints
  for (i in 1:N) {
    int shard = level[i]; // shards are indexed by the levels
    xi[shard, pos[shard]] = y[i];
    pos[shard] += 1;
  }
}

parameters {
  real mu;               // intercept
  real<lower = 0> sigma; // group-level standard deviation
  vector[L] alpha;       // unscaled group-level effects
}

model {
  // package up parameters for mapping
  vector[2] global = [mu, sigma]';

  vector[1] local[L];
  for (j in 1:L) {
    local[j] = [alpha[j]]';
  }

  // priors
  mu ~ normal(0, 1);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 0.5);

  // likelihood (via map-reduce)
  target += sum(
    map_rect(lp, global, local, xr, xi)
  );
}
