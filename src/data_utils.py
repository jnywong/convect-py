import h5py
import os

def save_inputs(saveDir,nz,nn,a,Ra,Pr,dt,nt,nout):
    # if os.path.isdir(saveDir) or mkpath(saveDir):
    saveName="{}{}".format(saveDir,"/inputs.h5")
    fid = h5py.File(saveName,"w")
    fid["nz"] = nz
    fid["nn"] = nn
    fid["a"] = a
    fid["Ra"] = Ra
    fid["Pr"] = Pr
    fid["dt"] = dt
    fid["nt"] = nt
    fid["nout"] = nout
    # close(fid)

def save_data(saveDir, ndata, dtemdt, domgdt, tem, omg, psi):
    # isdir(saveDir) or mkpath(saveDir)
    saveName="{}{}{:04}{}".format(saveDir,"/data_",ndata,".h5")
    fid = h5py.File(saveName,"w")
    fid["dtemdt"] = dtemdt
    fid["domgdt"] = domgdt
    fid["psi"] = psi
    fid["omg"] = omg
    fid["tem"] = tem
    # close(fid)

def save_outputs(saveDir,time,ndata):
    # isdir(saveDir) or mkpath(saveDir)
    saveName="{}{}".format(saveDir,"/outputs.h5")
    fid = h5py.File(saveName,"w")
    fid["time"] = time
    fid["ndata"] = ndata
    # close(fid)

def load_inputs(saveDir):
    saveName="{}{}".format(saveDir,"/inputs.h5")
    fid = h5py.File(saveName,"r")
    dset = fid["nz"]
    nz = dset[()]
    dset = fid["nn"]
    nn = dset[()]
    dset = fid["a"]
    a = dset[()]
    dset = fid["Ra"]
    Ra = dset[()]
    dset = fid["Pr"]
    Pr = dset[()]
    dset = fid["dt"]
    dt = dset[()]
    dset = fid["nt"]
    nt = dset[()]
    dset = fid["nout"]
    nout = dset[()]
    # close(fid)
    return nz,nn,a,Ra,Pr,dt,nt,nout

def load_data(saveDir, ndata):
    saveName="{}{}{:04}{}".format(saveDir,"/data_",ndata,".h5")
    fid = h5py.File(saveName,"r")
    dset = fid["dtemdt"]
    dtemdt = dset[()]
    dset = fid["domgdt"]
    domgdt = dset[()]
    dset = fid["psi"]
    psi = dset[()]
    dset = fid["omg"]
    omg = dset[()]
    dset = fid["tem"]
    tem = dset[()]
    # close(fid)
    return dtemdt, domgdt, tem, omg, psi

def load_outputs(saveDir):
    saveName="{}{}".format(saveDir,"/outputs.h5")
    fid = h5py.File(saveName,"r")
    dset = fid["time"]
    time = dset[()]
    dset = fid["ndata"]
    ndata = dset[()]
    # close(fid)
    return time, ndata