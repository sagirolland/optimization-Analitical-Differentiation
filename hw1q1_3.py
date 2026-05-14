import numpy as np
import hw1q1_2_3 as nummeric
import opt_hw1_5 as opt
import matplotlib.pyplot as plt

def norm_inf(v):
    return np.max(np.abs(v))

def comp_grad_diff(grad_analytical, grad_numerical):
    return norm_inf(grad_analytical - grad_numerical)

def comp_hess_diff(hess_analytical, hess_numerical):
    return norm_inf(hess_analytical - hess_numerical)

def compare_results():
    rng = np.random.default_rng(42)
    x   = rng.standard_normal(3)
    A   = rng.standard_normal((3, 3))
    exponents = np.arange(0,61)
    epi = 2.0**(-exponents.astype(float))
    v1, g1, H1 = opt.calc_f1(x, A, True, True, True)
    v2, g2, H2 = opt.calc_f2(x, True, True, True)
    err_grad_f1 = np.zeros(len(epi))
    err_hess_f1 = np.zeros(len(epi))
    err_grad_f2 = np.zeros(len(epi))
    err_hess_f2 = np.zeros(len(epi))
    for k, eps in enumerate(epi):
        v3, g3, H3 = nummeric.numerical_diff(opt.calc_f1, x, eps, A)
        v4, g4, H4 = nummeric.numerical_diff(opt.calc_f2, x, eps)
        err_grad_f1[k] = comp_grad_diff(g1, g3)
        err_hess_f1[k] = comp_hess_diff(H1, H3)
        err_grad_f2[k] = comp_grad_diff(g2, g4)
        err_hess_f2[k] = comp_hess_diff(H2, H4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Analytical vs Numerical Differentiation — Infinity Norm Error",fontsize=13)
    plots = [(err_grad_f1, "f1 — Gradient error"),(err_hess_f1, "f1 — Hessian error"),(err_grad_f2, "f2 — Gradient error"),(err_hess_f2, "f2 — Hessian error"),]    

    for ax, (errors, title) in zip(axes.flatten(), plots):
        ax.semilogy(exponents,errors,color="steelblue",linewidth=1.8,marker="o",markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Exponent k  (ε = 2⁻ᵏ)")
        ax.set_ylabel("‖error‖∞  (log scale)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("comparison_plots.png", dpi=150)
    plt.show()
    print("Plot saved to comparison_plots.png")
        

if __name__ == "__main__":
    compare_results()
    