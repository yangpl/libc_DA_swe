 # - plot_checks.py
 #  - gradcheck.log
 #  - adjcheck.log
 #  - check_report.png

 #  What the plot includes:

 #  - Gradient check: FD vs adjoint directional derivatives per trial
 #  - Gradient check relative error (log scale)
 #  - Adjoint dot-product check: LHS vs RHS per trial
 #  - Adjoint check relative error (log scale)

 #  To regenerate after new runs:

 #  ./gradcheck > gradcheck.log
 #  ./adjcheck 20 8 1e-4 > adjcheck.log
 #  python3 plot_checks.py

import re
import numpy as np
import matplotlib.pyplot as plt


def parse_gradcheck(path):
    trials, fd, adj, rel = [], [], [], []
    mean_rel = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(
                r"trial=(\d+)\s+f0=([-\deE+.]+)\s+fd=([-\deE+.]+)\s+adj=([-\deE+.]+)\s+rel=([-\deE+.]+)",
                line,
            )
            if m:
                trials.append(int(m.group(1)))
                fd.append(float(m.group(3)))
                adj.append(float(m.group(4)))
                rel.append(float(m.group(5)))
            mm = re.search(r"mean_rel=([-\deE+.]+)", line)
            if mm:
                mean_rel = float(mm.group(1))
    return np.array(trials), np.array(fd), np.array(adj), np.array(rel), mean_rel


def parse_adjcheck(path):
    trials, lhs, rhs, rel = [], [], [], []
    mean_rel = None
    nt = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mt = re.search(r"nt=(\d+)", line)
            if mt:
                nt = int(mt.group(1))
            m = re.search(r"trial=(\d+)\s+lhs=([-\deE+.]+)\s+rhs=([-\deE+.]+)\s+rel=([-\deE+.]+)", line)
            if m:
                trials.append(int(m.group(1)))
                lhs.append(float(m.group(2)))
                rhs.append(float(m.group(3)))
                rel.append(float(m.group(4)))
            mm = re.search(r"mean_rel=([-\deE+.]+)", line)
            if mm:
                mean_rel = float(mm.group(1))
    return np.array(trials), np.array(lhs), np.array(rhs), np.array(rel), mean_rel, nt


def main():
    g_trials, g_fd, g_adj, g_rel, g_mean = parse_gradcheck("gradcheck.log")
    a_trials, a_lhs, a_rhs, a_rel, a_mean, nt = parse_adjcheck("adjcheck.log")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(g_trials, g_fd, "o-", label="FD directional derivative")
    ax1.plot(g_trials, g_adj, "s--", label="Adjoint directional derivative")
    ax1.set_title("Gradient Check: FD vs Adjoint")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Directional Derivative")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.semilogy(g_trials, np.maximum(g_rel, 1e-12), "o-", color="tab:red")
    ax2.set_title(f"Gradient Check Relative Error (mean={g_mean:.3e})")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Relative Error (log)")
    ax2.grid(True, which="both", alpha=0.3)

    ax3.plot(a_trials, a_lhs, "o-", label="<Jv, lambda> (FD)")
    ax3.plot(a_trials, a_rhs, "s--", label="<v, JT lambda> (Adjoint)")
    ax3.set_title(f"Adjoint Dot-Product Check (nt={nt})")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("Inner Product")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.semilogy(a_trials, np.maximum(a_rel, 1e-12), "o-", color="tab:green")
    ax4.set_title(f"Adjoint Check Relative Error (mean={a_mean:.3e})")
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("Relative Error (log)")
    ax4.grid(True, which="both", alpha=0.3)

    fig.suptitle("SWE Adjoint and Gradient Consistency Checks", fontsize=15)
    fig.tight_layout()
    fig.savefig("check_report.png", dpi=160)
    print("Saved check_report.png")
    plt.show()

if __name__ == "__main__":
    main()
