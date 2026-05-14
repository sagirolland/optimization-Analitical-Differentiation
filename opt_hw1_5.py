import numpy as np

def phi(v):
    return np.sin(v[0]*v[1]*v[2])

def grad_phi(v):
    c=np.cos(v[0]*v[1]*v[2])
    p= np.array([v[1]*v[2],v[0]*v[2],v[0]*v[1]])
    return c*p

def hess_phi(v):
    s=np.sin(v[0]*v[1]*v[2])
    c=np.cos(v[0]*v[1]*v[2])
    p=np.array([v[1]*v[2],v[0]*v[2],v[0]*v[1]])
    Q=np.array([[0, v[2], v[1]],
                [v[2], 0, v[0]],
                [v[1], v[0], 0]])
    return c*Q -s*np.outer(p,p)

def h(v):
    return np.sqrt(1+v**2)

def dh(v):
    return v/h(v)

def ddh(v):
    return 1.0/h(v)**3
    
    
def calc_f1(x,A,compute_value,compute_derivative,compute_hessian):
    phi_Ax = None
    grad_phi_Ax = None
    hess_phi_Ax = None
    Ax=A@x
    if compute_value:
        phi_Ax=phi(Ax)
    if compute_derivative:
        grad_phi_Ax=A.T@grad_phi(Ax)
    if compute_hessian:
        hess_phi_Ax=A.T@hess_phi(Ax)@A
    return phi_Ax, grad_phi_Ax, hess_phi_Ax

def calc_f2(x,compute_value,compute_derivative,compute_hessian):
    h_phi_x = None
    dh_phi_x = None
    ddh_phi_x = None
    if compute_value:
        h_phi_x=h(phi(x))
    if compute_derivative:
        dh_phi_x=dh(phi(x))*grad_phi(x)
    if compute_hessian:
        ddh_phi_x=ddh(phi(x))*np.outer(grad_phi(x),grad_phi(x)) + dh(phi(x))*hess_phi(x)
    return h_phi_x, dh_phi_x, ddh_phi_x

