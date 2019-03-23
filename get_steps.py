from rdp_accountant import _compute_rdp, _compute_delta, compute_rdp
import numpy as np

def get_sigma(q, epoches, epsilon, delta): # sigma allows us to run 10 epoches
    steps_per_epoch=int(1./q)
    steps=epoches*steps_per_epoch
    sigma=1
    while(True):#get sigma in interge
        loss=compute_rdp(q, sigma, steps, np.arange(2, 2+512))
        if_delta, _ =_compute_delta(np.arange(2, 2+512), loss, epsilon)
        if(if_delta>delta):
            sigma+=1
        else:
            break
    
    for i in range(10):#get sigma in .1
        sigma-=.1
        loss=compute_rdp(q, sigma, steps, np.arange(2, 2+512))
        if_delta, _ =_compute_delta(np.arange(2, 2+512), loss, epsilon)
        if(if_delta>delta):
            sigma+=.1
            break
    
    for i in range(10):#get sigma in .01
        sigma-=.01
        loss=compute_rdp(q, sigma, steps, np.arange(2, 2+512))
        if_delta, _ =_compute_delta(np.arange(2, 2+512), loss, epsilon)
        if(if_delta>delta):
            sigma+=.01
            break 

    return sigma


    
            