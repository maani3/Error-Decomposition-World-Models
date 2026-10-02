# Follow-up experiments

```
==============================================================================================
E5  K-ROUND ACCUMULATION  (limited data per round; one entry changes per round)
  400 usable seeds, K=10 rounds, n_round=25

  --- horizon h=1 ---
  rule        round-1 ENDPT  round-1 MISMCH   mean MISMCH   FINAL ENDPT (K)
  refit               0.177           1.301         1.560             0.158
  replay              0.152           1.112         0.975             0.357
  proximal            0.126           0.779         0.819             0.129
  adapter             0.101           0.184         0.199             0.112
  ranking by round-1 endpoint : ['adapter', 'proximal', 'replay', 'refit']
  ranking by round-1 mismatch : ['adapter', 'proximal', 'replay', 'refit']
  ranking by FINAL endpoint   : ['adapter', 'proximal', 'refit', 'replay']
  round-1 endpoint predicts final? NO   round-1 mismatch predicts final? NO
  Per-seed agreement with the final ordering:
    round1_endpoint   pairwise=0.755  best=0.637  mean Spearman=0.617
    round1_mismatch   pairwise=0.777  best=0.642  mean Spearman=0.675
    mean_mismatch     pairwise=0.740  best=0.637  mean Spearman=0.619
    mismatch - endpoint: pairwise +0.022 95% CI [+0.010, +0.035]; Spearman +0.058 [+0.034, +0.084]

  --- horizon h=5 ---
  rule        round-1 ENDPT  round-1 MISMCH   mean MISMCH   FINAL ENDPT (K)
  refit               0.481           1.779         2.805             0.463
  replay              0.320           1.584         1.055             0.746
  proximal            0.343           0.942         1.157             0.334
  adapter             0.375           0.332         0.348             0.328
  ranking by round-1 endpoint : ['replay', 'proximal', 'adapter', 'refit']
  ranking by round-1 mismatch : ['adapter', 'proximal', 'replay', 'refit']
  ranking by FINAL endpoint   : ['adapter', 'proximal', 'refit', 'replay']
  round-1 endpoint predicts final? NO   round-1 mismatch predicts final? NO
  Per-seed agreement with the final ordering:
    round1_endpoint   pairwise=0.510  best=0.260  mean Spearman=0.022
    round1_mismatch   pairwise=0.708  best=0.475  mean Spearman=0.496
    mean_mismatch     pairwise=0.596  best=0.470  mean Spearman=0.236
    mismatch - endpoint: pairwise +0.198 95% CI [+0.173, +0.223]; Spearman +0.474 [+0.415, +0.531]

==============================================================================================
E6  NONLINEAR STRESS TEST  (x_{t+1} = A tanh(x_t); rollout is nonlinear)
  399/400 usable, median inherited error h=1 = 0.091
  rule         ENDPT h=1   ENDPT h=4   ENDPT h=8  |  MISMCH h=1  MISMCH h=4  MISMCH h=8
  refit            0.025       0.054       0.095  |       0.422       0.778       0.948
  replay           0.111       0.190       0.271  |       0.632       0.915       1.111
  adapter          0.088       0.281       0.492  |       0.037       0.223       0.449
  rank disagreement:   h=1 100.0%  h=2 100.0%  h=4  96.7%  h=8  75.9%

==============================================================================================
E7  AGGREGATION  (worst-case column vs global Frobenius error)
  545 usable seeds
  rule        GLOBAL h=1  WORST h=1   ratio  GLOBAL h=12  WORST h=12
  refit            0.470      0.296    0.63        1.070       1.034
  replay           0.661      0.533    0.81        1.213       1.217
  adapter          0.047      0.047    1.00        0.667       0.664

==============================================================================================
E8  WILSON 95% CIs ON HEADLINE DISAGREEMENT RATES
  inherited-bias base    h1     100.0%   95% CI [ 99.3, 100.0]  n=544
  inherited-bias base    h5      93.8%   95% CI [ 91.4,  95.5]  n=544
  inherited-bias base    h12     66.2%   95% CI [ 62.1,  70.0]  n=544
  low-error base         h1       0.0%   95% CI [  0.0,   0.7]  n=544
  low-error base         h5       2.4%   95% CI [  1.4,   4.0]  n=544
  low-error base         h12      5.9%   95% CI [  4.2,   8.2]  n=544

==============================================================================================
E9  NON-ORACLE PROXIMAL UPDATE ABLATION
  544 usable seeds
      mu    disagree h=1    disagree h=5   disagree h=12
       1           20.4%           33.1%           29.4%
       5           24.6%           34.4%           31.4%
      10           31.4%           36.6%           32.0%
      20           45.4%           39.5%           33.1%
      40           70.8%           44.9%           35.1%
      80           85.8%           54.4%           40.1%
     160           67.5%           62.5%           45.8%

==============================================================================================
E10  CANCELLATION GEOMETRY  (signed identity; proximal mu=80)
  544 usable seeds
  rule          cos h=1    cos h=5   cos h=12  |  endpoint/tri h=1  endpoint/tri h=5 endpoint/tri h=12
  refit          -0.929     -0.979     -0.986  |             0.196             0.132             0.169
  proximal       -0.859     -0.958     -0.971  |             0.270             0.189             0.221
  adapter        -0.079     -0.060     -0.144  |             0.893             0.811             0.798
```
