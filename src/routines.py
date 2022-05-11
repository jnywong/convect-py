
import os
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from shutil import rmtree

import src.routines as routines
from src.data_utils import save_inputs, save_data, save_outputs, load_inputs, load_data, load_outputs

def poisson(b,nz,dz,n,a):
    d = np.zeros(nz) # diagonal
    du = -1/dz**2*np.ones(nz-1) # upper diagonal
    dl = -1/dz**2*np.ones(nz-1) # lower diagonal
    for k in range(1, nz-1): # 2:1:nz-1
        d[k] = (n*np.pi/a)**2 + 2/dz**2
    
    # Boundary conditions
    d[0] = 1 # (2.22a)
    du[0] = 0 #(2.22b)
    b[0] = 0 # (2.22c)
    dl[nz-2] = 0 # (2.22d)
    d[nz-1] = 1 # (2.22e)
    b[nz-1] = 0 # (2.22f)
    diagonals = [dl,d,du]
    A = diags(diagonals, [-1,0,1], format='csc')
    x = spsolve(A, b)
    return x

def xdomain(a, nz):
    nx = int(ceil(a*nz)) # no. of horizontal gridpoints
    dx = a/(nx-1)
    x = np.linspace(0.0,a,nx)
    return x, dx, nx

def zdomain(nz):
    z = np.linspace(0,1,nz)
    dz = 1/(nz-1)
    return z, dz

def preallocate_spat(nz, nx):
    tem = np.zeros(nz, nx)
    psi = np.zeros(nz, nx)
    omg = np.zeros(nz, nx)
    return tem, omg, psi

def preallocate_spec(nz, nn):
    # Includes zeroth mode
    psi = np.zeros((nz,nn,2))
    tem = np.zeros((nz,nn,2))
    omg = np.zeros((nz,nn,2))
    return psi, tem, omg 

def initial_linear_tem(nz,nn,z,tem):
    tem[:,0,2] = np.ones(nz) - z # zeroth mode
    # tem[:,2,2] = sin.(pi*z)
    for n in range(1,nn): #n=2:1:nn+1
        tem[:,n,2] = np.sin(np.pi*z) # satisfies (3.6) T(0)=T(1) at time t=0
    return tem

def initial_nonlinear_tem(nz,nn,z,tem,fac=1e-3):
    for i in range(0,2): #i=1:1:2
        tem[:,0,i] = np.ones(nz) - z # zeroth mode
        tem[:,1,i] = 0.01*np.sin(np.pi*z) # §4.4.2 A nonlinear benchmark
    
    # for n=2:1:nn+1
    #     tem[:,n,2] = fac*rand(Uniformtree(-1, 1))*sin.(pi*z) # satisfies (3.6) T(0)=T(1) at time t=0
    # 
    return tem

def ref_tem(nx,nz,z,tem_out):
    for i in range(0,nx): #i=1:1:nx
        tem_out[:,i] = np.ones(nz) - z # reference temperature
    return tem_out

def first_deriv(k,n,dz,y,dydz):
    # First derivative w.r.t z
    dydz[k,n] = (y[k+1,n,1] - y[k-1,n,1])/(2*dz) # (2.15)
    return dydz

def second_deriv(k,n,dz,y,dydz2):
    # Second derivative w.r.t z
    dydz2[k,n] = (y[k+1,n,1] - 2*y[k,n,1] + y[k-1,n,1])/(dz**2) # (2.16)
    return dydz2

def lin_tem_eq(k,n,a,tem,psi,dtemdz2,dtemdt):
    # Update time deriv dtemdt
    dtemdt[k,n,1] = (n*np.pi/a)*psi[k,n,1] + (dtemdz2[k,n]-(n*np.pi/a)**2*tem[k,n,1]) # (3.3)
    return dtemdt

def lin_omg_eq(k,n,a,Ra,Pr,tem,omg,domgdz2,domgdt):
    # Update time deriv domgt
    domgdt[k,n,1] = Ra*Pr*(n*np.pi/a)*tem[k,n,1] + Pr*(domgdz2[k,n] - (n*np.pi/a)**2*omg[k,n,1]) # (3.4)
    return domgdt 

def nonlin_tem_eq(k,n,a,tem,psi,dtemdz2,dtemdt):
    # Update time deriv dtemdt
    dtemdt[k,n,1] = (dtemdz2[k,n]-(n*np.pi/a)**2*tem[k,n,1]) # (3.3) with first term on RHS removed
    return dtemdt

def nonlin_omg_eq(k,n,a,Ra,Pr,tem,omg,domgdz2,domgdt):
    # Update time deriv domgt
    domgdt[k,n,1] = Ra*Pr*(n*np.pi/a)*tem[k,n,1] + Pr*(domgdz2[k,n] - (n*np.pi/a)**2*omg[k,n,1]) # same as (3.4)
    return domgdt 

def adamsbashforth(n, y, dydt, dt):
    y[:,n,1] = y[:,n,0] + 0.5*dt*(3*dydt[:,n,1]-dydt[:,n,0]) # (2.18)
    return y

def diagnostics(m,nz,nout,time,tem,omg,psi):
    # track n=1 mode over time at z = nz/3
    index = int(np.round(nz/3))
    print("time: {:.2f}    tem: {:.5e}    omg: {:.5e}    psi: {:.5e} \n ".format(time, np.log(abs(tem[index,2,1]))-np.log(abs(tem[index,2,0])), np.log(abs(omg[index,2,1]))-np.log(abs(omg[index,2,0])),np.log(abs(psi[index,2,1]))-np.log(abs(psi[index,2,0]))))

def prepare(dtemdt, domgdt, tem, omg, psi):
    dtemdt[:,:,0] = dtemdt[:,:,1]
    domgdt[:,:,0] = domgdt[:,:,1]
    tem[:,:,0] = tem[:,:,1]
    omg[:,:,0] = omg[:,:,1]
    psi[:,:,0] = psi[:,:,1]
    dtemdt[:,:,1] = 0
    domgdt[:,:,1] = 0 
    return dtemdt, domgdt, tem, omg, psi

def linear_solver(z, dz, nz, nn, nt, nout, dt, a, Ra, Pr, psi, tem, omg, initOn, saveDir):
    if initOn==1:
        if os.path.isdir(saveDir)==1:
            rmtree(saveDir)
        os.mkdir(saveDir)
        save_inputs(saveDir,nz,nn,a,Ra,Pr,dt,nt,nout)
        ndata = 0 
        time = 0
        dtemdt = np.zeros(tem.shape)
        domgdt = np.zeros(omg.shape)
        tem = initial_linear_tem(nz,nn,z,tem)
    elif initOn==0:
        nz,nn,a,Ra,Pr,dt,nt,nout = load_inputs(saveDir)
        time, ndata = load_outputs(saveDir)
        dtemdt, domgdt, tem, omg, psi = load_data(saveDir,ndata-1)
    
    m = 0
    time = 0
    dtemdz2 = np.zeros(nz,nn)
    domgdz2 = np.zeros(nz,nn)

    while m<=nt:
        for k in range(1, nz-1): #k=2:1:nz-1 # loop over interior z
            for n in range(1,nn):# n=2:1:nn+1 # loop over Fourier modes
                # println("n=",n-1)
                dtemdz2 = second_deriv(k ,n, dz, tem, dtemdz2)
                domgdz2 = second_deriv(k ,n, dz, omg, domgdz2)
                dtemdt = lin_tem_eq(k,n,a,tem,psi,dtemdz2,dtemdt)
                domgdt = lin_omg_eq(k,n,a,Ra,Pr,tem,omg,domgdz2,domgdt)
            
        # Update tem and omg using Adams Bashforth time integration
        for n in range(1,nn): #n=2:1:nn+1
            tem = adamsbashforth(n, tem, dtemdt, dt)
            omg = adamsbashforth(n, omg, domgdt, dt)

        # Update psi using poisson solver
        for n in range(1,nn): # n=2:1:nn+1
            psi[:,n,2] = poisson(omg[:,n,1],nz,dz,n,a) # (3.5)

        # Diagnostics
        if  m % nout == 0: #mod(m,nout)==0
            diagnostics(m, nz, nout, time, tem, omg, psi)
            save_data(saveDir,ndata,dtemdt, domgdt, tem, omg, psi)
            ndata+=1
            # Prepare values for next timestep
            dtemdt, domgdt, tem, omg, psi = prepare(dtemdt, domgdt, tem, omg, psi)
            # open(string(saveDir,"/k_33_dtemdt.csv"),"a+") do f
            # @printf(f,"%16.8f\n",dtemdt[33,2,2])
            # 
            # open(string(saveDir,"/k_33_tem.csv"),"a+") do f
            # @printf(f,"%16.8f\n",tem[33,2,2])
            # 
        else:
            # Prepare values for next timestep
            dtemdt, domgdt, tem, omg, psi = prepare(dtemdt, domgdt, tem, omg, psi)
        
        m+=1
        time += dt
        save_outputs(saveDir, time, ndata)  
    return dtemdt, domgdt, tem, omg, psi

def nonlinear_solver(z, dz, nz, nn, nt, nout, dt, a, Ra, Pr, psi, tem, omg, initOn, saveDir):
    if initOn==1:
        if os.path.isdir(saveDir)==1:
            rmtree(saveDir)
        os.mkdir(saveDir)
        save_inputs(saveDir,nz,nn,a,Ra,Pr,dt,nt,nout)
        ndata = 0
        time = 0
        dtemdt = np.zeros(tem.shape)
        domgdt = np.zeros(omg.shape)
        tem = routines.initial_nonlinear_tem(nz,nn,z,tem)
    elif initOn==0:
        nz,nn,a,Ra,Pr,dt,nt,nout = load_inputs(saveDir)
        time, ndata = load_outputs(saveDir)
        dtemdt, domgdt, tem, omg, psi = load_data(saveDir,ndata-1)
    
    m = 0
    dtemdz1 = np.zeros((nz,nn))
    domgdz1 = np.zeros((nz,nn)) 
    dpsidz1 = np.zeros((nz,nn))
    dtemdz2 = np.zeros((nz,nn))
    domgdz2 = np.zeros((nz,nn))
    while m<nt:
        for k in range(1, nz-2): #k=2:1:nz-1
            # if k==33
            #     println("k:", k)
            # 
            # Linear terms
            for n in range(0,nn): #n=1:1:nn+1
                # print(k,n)
                dtemdz1 = first_deriv(k ,n, dz, tem, dtemdz1)
                domgdz1 = first_deriv(k ,n, dz, omg, domgdz1)
                dpsidz1 = first_deriv(k ,n, dz, psi, dpsidz1)
                dtemdz2 = second_deriv(k ,n, dz, tem, dtemdz2)
                domgdz2 = second_deriv(k ,n, dz, omg, domgdz2)
                dtemdt = nonlin_tem_eq(k, n, a, tem, psi, dtemdz2, dtemdt)
                domgdt = nonlin_omg_eq(k, n, a, Ra, Pr, tem, omg, domgdz2, domgdt)
            
            # Nonlinear terms
            for n1 in range(1, nn): #n1=2:1:nn+1
                # Zeroth mode
                dtemdt[k,0,1] += -np.pi/(2*a)*n1*(dpsidz1[k,n1]*tem[k,n1,0]+psi[k,n1,1]*dtemdz1[k,n1])
            
            for n in range(1,nn): #n=2:1:nn+1
                # if n==3
                #     println("n:", n-1)
                # 
                # n'= 0 mode
                dtemdt[k,n,1] += -n*np.pi/a*psi[k,n,1]*dtemdz1[k,0]
                # 0 < n' < nn
                for n1 in range(1, nn): #n1=2:1:nn+1
                    # println("n1:", n1-1)
                    n2 = np.zeros(3)
                    tem_term = np.zeros(3)
                    omg_term = np.zeros(3)
                    n2[0] = n-n1
                    n2[1] = n+n1
                    n2[2] = n1-n
                    for i in range(0,n2.size): #i=1:1:length(n2)
                        # Check if 1<=n<=nn, no contribution if not
                        if i==0 and n2[i]>=1 and n2[i]<=nn-1:
                            tem_term[i] = -n1*dpsidz1[k,int(n2[i])]*tem[k,n1,1]+n2[i]*psi[k,int(n2[i]),1]*dtemdz1[k,n1]
                            omg_term[i] = -n1*dpsidz1[k,int(n2[i])]*omg[k,n1,1]+n2[i]*psi[k,int(n2[i]),1]*domgdz1[k,n1]
                        elif i==1 and n2[i]>=1 and n2[i]<=nn-1:
                            tem_term[i] = n1*dpsidz1[k,int(n2[i])]*tem[k,n1,1]+ n2[i]*psi[k,int(n2[i]),1]*dtemdz1[k,n1]
                            omg_term[i] = -n1*dpsidz1[k,int(n2[i])]*omg[k,n1,1]-n2[i]*psi[k,int(n2[i]),1]*domgdz1[k,n1]
                        elif i==2 and n2[i]>=1 and n2[i]<=nn-1:
                            tem_term[i] = n1*dpsidz1[k,int(n2[i])]*tem[k,n1,1]+n2[i]*psi[k,int(n2[i]),1]*dtemdz1[k,n1]
                            omg_term[i] = n1*dpsidz1[k,int(n2[i])]*omg[k,n1,1]+n2[i]*psi[k,int(n2[i]),1]*domgdz1[k,n1]
                        
                    dtemdt[k,n,1] += -np.pi/(2*a)*(np.sum(tem_term))
                    domgdt[k,n,1] += -np.pi/(2*a)*(np.sum(omg_term))
                    # if sum(tem_term)!=0
                    #     println(sum(tem_term))
                    # 
        # display(dtemdt[33,:,:])

        # Update tem and omg using Adams Bashforth time integration
        for n in range(0,nn): #n=1:1:nn+1
            tem = adamsbashforth(n, tem, dtemdt, dt)
            omg = adamsbashforth(n, omg, domgdt, dt)
        
        # Update psi using poisson solver
        for n in range(1,nn): # n=2:1:nn+1
            psi[:,n,1] = poisson(omg[:,n,1],nz,dz,n,a) # (3.5)
        
        # Save and print diagnostics
        if m !=0 and m % nout ==0:
            diagnostics(m, nz, nout, time, tem, omg, psi)
            save_data(saveDir,ndata,dtemdt, domgdt, tem, omg, psi)
            ndata+=1
            # open(string(saveDir,"/k_33_n_1_dtemdt.csv"),"a+") do f
            # @printf(f,"%16.8f\n",dtemdt[33,2,2])
            # 
            # open(string(saveDir,"/k_33_n_1_tem.csv"),"a+") do f
            # @printf(f,"%16.8f\n",tem[33,2,2])
            #         
            # open(string(saveDir,"/k_33_n_2_dtemdt.csv"),"a+") do f
            # @printf(f,"%16.8f\n",dtemdt[33,3,2])
            # 
            # open(string(saveDir,"/k_33_n_2_tem.csv"),"a+") do f
            # @printf(f,"%16.8f\n",tem[33,3,2])
            #         
            # Prepare values for next timestep
            dtemdt, domgdt, tem, omg, psi = prepare(dtemdt, domgdt, tem, omg, psi)
        else:
            # Prepare values for next timestep
            dtemdt, domgdt, tem, omg, psi = prepare(dtemdt, domgdt, tem, omg, psi)
        
        m+=1
        time += dt
        save_outputs(saveDir, time, ndata)
    
    return dtemdt, domgdt, tem, omg, psi

def cosines(a, x, nn, nx):
    # Compute cosines and sines 
    cosa = np.zeros(nn,nx)
    for n in range(0,nn): #n=1:1:nn+1
        cosa[n,:] = np.cos(n*np.pi*x/a)
    return cosa

def sines(a, x, nn, nx):
    # Compute cosines and sines 
    sina = np.zeros(nn,nx)
    for n in range(0,nn): #n=1:1:nn+1
        sina[n,:] = np.sin(n*np.pi*x/a)
    return sina

def ict(nn,nx,nz,cosa,coeffs,outfun,zeroth=1):
    for i in range (0,nx-1): #i = 1:1:nx
        for k in range(0,nz-1): #k = 1:1:nz
            if zeroth==1: # include zeroth mode
                for n in range(0,nn): #n=1:1:nn+1
                    outfun[k,i] += coeffs[k,n,1]*cosa[n,i]  
            elif zeroth==0: # exclude zeroth mode
                for n in range(1,nn): #n=2:1:nn+1
                    outfun[k,i] += coeffs[k,n,1]*cosa[n,i]
    return outfun

def ist(nn,nx,nz,sina,coeffs,outfun):
    for i in range(0, nx-1): #i = 1:1:nx
        for k in range(0, nz-1): #k = 1:1:nz
            for n in range(0,nn): #n=1:1:nn+1
                outfun[k,i] += coeffs[k,n,1]*sina[n,i]
    return outfun
