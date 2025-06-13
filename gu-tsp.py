import numpy as np

# --- 1. model wrapper -------------------------------------------------
def run_emt_sim(Kp, Ki, line_model):
    """
    Black-box time-domain simulation of the entire feeder.
    Returns aggregate cost: J = w1*settling + w2*overshoot + w3*RMS_error + w4*quench_risk
    """
    # call out to PSCAD/EMTP, Modelica, or your home-grown solver
    # NOTE: make the wrapper GPU-friendly if the solver is PyTorch-based
    return J

# --- 2. elastic gradient descent -------------------------------------
def elastic_tune(Kp, Ki, field_grad, steps=300, alpha=0.4, beta=0.1, decay=0.995):
    N = len(Kp)
    for t in range(steps):
        gKp, gKi = field_grad(Kp, Ki)         # shape (N,) each
        lapKp = np.roll(Kp,1)+np.roll(Kp,-1)-2*Kp    # Laplacian smooth term
        lapKi = np.roll(Ki,1)+np.roll(Ki,-1)-2*Ki
        Kp -= alpha*gKp + beta*lapKp
        Ki -= alpha*gKi + beta*lapKi
        alpha*=decay;  beta*=decay*0.9
    return Kp, Ki

# --- 3. automatic gradient via finite diff (replace with autograd if possible)
def field_grad(Kp, Ki, h=1e-3):
    base = run_emt_sim(Kp, Ki, line_model)
    gKp = np.zeros_like(Kp);  gKi = np.zeros_like(Ki)
    for i in range(len(Kp)):
        Kp[i] += h;  gKp[i] = (run_emt_sim(Kp,Ki,line_model)-base)/h;  Kp[i]-=h
        Ki[i] += h;  gKi[i] = (run_emt_sim(Kp,Ki,line_model)-base)/h;  Ki[i]-=h
    return gKp, gKi