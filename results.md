# Experiment results

```
IDENTITY CHECK
  max |(delta_M - delta_E) - (eps^1 - eps^0)| over h=1..12  =  2.08e-17

==============================================================================================
EXPERIMENT A1 -- INHERITED-BIAS BASE  (ridge lam=40, n=400)
  seeds: 600 attempted, 544 usable (90.7%)   rejected: rho_A1_too_large=54, change_does_not_propagate=2
  median inherited error BASE(h=1) = 0.100
            ENDPT h=1   ENDPT h=5  ENDPT h=12  |  MISMCH h=1  MISMCH h=5 MISMCH h=12
refit           0.039       0.105       0.230  |       0.470       0.935       1.069
replay          0.115       0.214       0.373  |       0.661       1.086       1.213
adapter *       0.096       0.379       0.682  |       0.047       0.336       0.666
  * adapter = ORACLE DIAGNOSTIC, not a proposed algorithm
  RANK DISAGREEMENT (refit better on ENDPOINT and adapter better on MISMATCH):
      h=1  100.0%  h=2  100.0%  h=3   99.4%  h=5   93.8%  h=8   80.9%  h=12  66.2%

==============================================================================================
EXPERIMENT A2 -- LOW-ERROR BASE      (OLS, n=8000)
  seeds: 600 attempted, 544 usable (90.7%)   rejected: rho_A1_too_large=54, change_does_not_propagate=2
  median inherited error BASE(h=1) = 0.009
            ENDPT h=1   ENDPT h=5  ENDPT h=12  |  MISMCH h=1  MISMCH h=5 MISMCH h=12
refit           0.039       0.104       0.241  |       0.179       0.259       0.388
replay          0.115       0.218       0.377  |       0.517       0.559       0.625
adapter *       0.010       0.028       0.061  |       0.023       0.039       0.066
  * adapter = ORACLE DIAGNOSTIC, not a proposed algorithm
  RANK DISAGREEMENT (refit better on ENDPOINT and adapter better on MISMATCH):
      h=1    0.0%  h=2    0.0%  h=3    0.4%  h=5    2.4%  h=8    3.5%  h=12   5.9%

==============================================================================================
EXPERIMENT B -- BASE-ERROR SWEEP  (lam in 0,5,10,20,40,80 at n=400; plus n=8000)
   lam     n0  usable   BASE(1)   dis h=1   dis h=3   dis h=5  dis h=12
     0    400     544     0.040     53.1%     52.4%     48.7%     36.8%
     5    400     544     0.041     59.7%     56.2%     52.8%     39.9%
    10    400     544     0.046     73.9%     73.2%     64.5%     45.6%
    20    400     544     0.062     96.3%     95.2%     83.1%     58.1%
    40    400     544     0.100    100.0%     99.4%     93.8%     66.2%
    80    400     544     0.172    100.0%     99.6%     95.2%     67.5%
     0   8000     544     0.009      0.0%      0.4%      2.4%      5.9%

==============================================================================================
EXPERIMENT C -- SPECTRAL SWEEP  (adapter MISMATCH vs horizon, inherited-bias base)
 rho(A0)  usable      h=1      h=2      h=3      h=5      h=8     h=12   growth
    0.70     544    0.047    0.111    0.188    0.336    0.505    0.666     14.2x
    0.90     549    0.053    0.110    0.184    0.331    0.505    0.664     12.6x
    0.95     549    0.054    0.111    0.184    0.331    0.505    0.664     12.4x
    1.02     551    0.056    0.111    0.184    0.330    0.504    0.665     11.8x
```
