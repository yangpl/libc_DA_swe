• Current code is a DTA-only 2D SWE 4D-Var pipeline:

  - Forward model:
      - swe_step_forward in swe.c
      - stores trajectory (z/u/v_store) each time step for adjoint replay
  - Adjoint model (discrete adjoint):
      - swe_step_adjoint in swe.c
      - uses two-buffer logic (az_in/au_in/av_in snapshots + accumulators) for stable reverse propagation
  - Cost + gradient:
      - compute_cost in swe.c
      - cost is sampled-observation misfit over time
      - gradient is initial-time a_z after full backward sweep
  - Optimization:
      - NLCG (Polak-Ribiere) in main.c
      - Wolfe-style line search with NaN guards and descent-direction reset
      - outputs: z_true_init.bin, z_guess_init.bin, z_final_analysis.bin, cost_history.txt
  - Key fixes already integrated:
      - correct sampled-point count np
      - boundary-safe temp field initialization in forward step
      - removed legacy adjoint branch; only one DTA adjoint remains
      - run script fixed to plot_result.py
  - Verification tools:
      - gradient check: gradcheck.c
      - multistep adjoint dot-product check: adjcheck.c
      - plotting checks: plot_checks.py -> check_report.png
      
• Compile:

  make

  Run 4D-Var:

  ./main

  Optional parameters:

  ./main niter=100 nt=100 nx=101 ny=101 dt=0.1 dx=10 dy=10 stride_x=5 stride_y=5 initopt=0

  Main outputs:

  - z_true_init.bin
  - z_guess_init.bin
  - z_final_analysis.bin
  - cost_history.txt

  Visualize assimilation result:

  python3 plot_result.py

  Generates:

  - 4dvar_report.png

  Run consistency checks:

  make gradcheck adjcheck
  ./gradcheck > gradcheck.log
  ./adjcheck 20 8 1e-4 > adjcheck.log

  Visualize check diagnostics:

  python3 plot_checks.py

  Generates:

  - check_report.png

