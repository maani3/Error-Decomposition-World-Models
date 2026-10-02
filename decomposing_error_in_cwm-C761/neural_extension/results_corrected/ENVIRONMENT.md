# Corrected neural experiment environment

- Run date: 2026-08-30
- Command: `python run_corrected.py full`
- Python: 3.12.3
- PyTorch: 2.3.1+cu121
- NumPy: 1.26.4
- PyYAML: 6.0.1
- Matplotlib: 3.6.3
- Device: CPU (`torch.cuda.is_available() == False`)
- Workers: 10 single-threaded processes
- Wall time: 719 seconds
- Attempted/retained: 500/500 condition records across 100 paired seeds
- Raw result: `results_v1.pkl`

The corrected configuration is stored both in `../config_corrected.yaml` and
inside the raw pickle.  All five N0 conditions for a seed share D1 and use nested D0 prefixes.
