import pandas as pd
import numpy as np
import os.path
import sys
from scipy.spatial.transform import Rotation
import coordinatesConvertor_scipy as Ccvtor

def readTUM(arg):
    df = pd.read_csv(arg, header=None, delim_whitespace=True)

    # timestamp tx ty tz qx qy qz qw
    time = df[0].tolist() 
    tx   = df[1].tolist()
    ty   = df[2].tolist()
    tz   = df[3].tolist()

    qx   = df[4].tolist() 
    qy   = df[5].tolist()
    qz   = df[6].tolist()
    qw   = df[7].tolist()

    tx_o = []
    ty_o = []
    tz_o = []

    qx_o = []
    qy_o = []
    qz_o = []
    qw_o = []

    for idx in range(len(time)):
        # coordinate system y forward
        #rtmp = Rotation.from_euler('yxz', [ry[idx], rx[idx], rz[idx]], degrees=True)
        # coordinate system z forward
        rtmp = Rotation.from_quat( [qx[idx], qy[idx], qz[idx], qw[idx]])
        ttmp = np.matrix([tx[idx], ty[idx], tz[idx]])
        ro, to = Ccvtor.coordCvtor.from_Unity3D(rtmp, ttmp).to_SLAM()
        qx_o.append(ro.as_quat()[0])
        qy_o.append(ro.as_quat()[1])
        qz_o.append(ro.as_quat()[2])
        qw_o.append(ro.as_quat()[3])
        tx_o.append(to[0,0])
        ty_o.append(to[0,1])
        tz_o.append(to[0,2])


    o_df = pd.DataFrame([])
    o_df = o_df.append(pd.DataFrame({'time': time, 'x': tx_o, 'y': ty_o, 'z': tz_o, 'q0': qx_o, 'q1': qy_o, 'q2': qz_o, 'q3': qw_o}))

    filename_w_ext = os.path.basename(arg)
    filename, file_extension = os.path.splitext(filename_w_ext)
    o_df.to_csv(os.path.dirname(os.path.abspath(arg))+'/'+filename+'_converted'+file_extension, sep=' ', index=False, header=False, columns=['time', 'x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'])



def main(arg):
    # the freq of the sampling

    lenArg = len(arg)
    print(lenArg," and ",arg) 

    for i in range(0,lenArg):
        readTUM(arg[i])

    return
if __name__ == "__main__":
    main(sys.argv[1:])
