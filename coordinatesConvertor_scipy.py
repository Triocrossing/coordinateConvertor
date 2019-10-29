import numpy as np
import math
#from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from enum import Enum
import sys

# Convert world coordiantes to relative coordiantes
# ALL the inputs are reprensented in the WORLD FRAME

'''
T0: PREVIOUS translation vector: 
T0 = [t0, t1, t2]  
T1: CURRENT translation vector:  
T1 = [t0, t1, t2]  

Q0: PREVIOUS quaternion vector:  
Q0 = Rotation [x=q0, y=q1, z=q2, w=q3]
Q1: CURRENT quaternion vector:
Q1 = Rotation [x=q0, y=q1, z=q2, w=q3]

return:
rT1, rq1: RELATIVE pose by taking t0 and Q0 as origin
'''

def world2Relative(t0, t1, Q0, Q1):

    assert(np.shape(t0)==np.shape(t1)==(3,))

    # Get rotation matrix
    r0 = Q0.as_dcm()
    r1 = Q1.as_dcm()

    # Calculate rQ1
    rq1 = Q1*(Q0.inv())


    # Calculate rT1
    m_r1r0_t=np.dot(-r1,np.transpose(r0))
    rT1 = np.dot(m_r1r0_t,t0)+t1

    # Since relative T0 and R0 are null elements
    return rT1, rq1 


# Convert world coordiantes to relative coordiantes
'''
T0: PREVIOUS translation vector IN WORLD(ACCUMULATED):    
T0 = [t0, t1, t2]  
T1: CURRENT translation vector IN RELATIVE:               
T1 = [t0, t1, t2]  

Q0: PREVIOUS quaternion vector IN WORLD(ACCUMULATED):     
Q0 = Quaternion[x=q0, y=q1, z=q2, w=q3]
Q1: CURRENT quaternion vector IN RELATIVE:                
Q1 = Quaternion[x=q0, y=q1, z=q2, w=q3]
'''

def relative2World(t0, t1, Q0, Q1):

    assert(np.shape(t0)==np.shape(t1)==(3,1))

    # Get rotation matrix
    r1 = Q1.as_dcm()

    # Calcualte rQ1
    wq1 = Q1*Q0

    # Calculate rT1
    wT1 = np.dot(r1,t0)+t1

    return wq1, wT1


# A Class converts different coordinate defintions

''' 

without any indication, the class takes z-forward (** SLAM ** method) 
as default coordinate system while all converting process

supported types: 
OpenGL, Unity3D, FreeD, Unreal, SLAM

By Xi WANG 29/10/2019

'''

class CoordType(Enum):
  SLAM    = 0
  OpenGL  = 1
  Unity3D = 2
  FreeD   = 3
  UE      = 4

# precomputed transform matrices
s_M_opgl = np.matrix([[1,0,0],[0,-1,0],[0,0,-1]])
s_M_u3d  = np.matrix([[1,0,0],[0,-1,0],[0,0,1]]) 
s_M_fd   = np.matrix([[1,0,0],[0,0,1],[0,-1,0]]) 
s_M_ue   = np.matrix([[0,0,1],[1,0,0],[0,-1,0]]) 

class coordCvtor():

  # R is Rotation in SciPy
  # t is the Matrix/Array in Numpy
  def __init__(self, R, t, cType=CoordType.SLAM):

    if(cType==CoordType.SLAM):
      self.R = R
      self.t = t
      return

    R_cvt, t_cvt = self.convert(R,t,cTypeIn=cType)
    self.R = R_cvt
    self.t = t_cvt

  '''
    Ctors
  '''
  @classmethod
  def from_SLAM(cls, R, t):
    return cls(R,t)

  @classmethod
  def from_OpenGL(cls, R, t):
    return cls(R, t, CoordType.OpenGL)

  @classmethod
  def from_Unity3D(cls, R, t):
    return cls(R, t, CoordType.Unity3D)

  @classmethod
  def from_FreeD(cls, R, t):
    return cls(R, t, CoordType.FreeD)

  @classmethod
  def from_UE(cls, R, t):
    return cls(R, t, CoordType.UE)

  def to_SLAM(self):
    return self.R, self.t

  def to_OpenGL(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.OpenGL)

  def to_Unity3D(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.Unity3D)

  def to_FreeD(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.FreeD)

  def to_UE(self):
    return self.convert(self.R, self.t, cTypeOut=CoordType.UE)

  '''
    Computation
  '''
  # generate transform matrix of specific in and out types
  @staticmethod
  def computeM_in_out(cTypeIn, cTypeOut):
    if(cTypeIn==cTypeOut):
      return np.eye(3) # eye mat

    # Choose correct s^M_in 
    if(cTypeIn==CoordType.SLAM):
      s_M_in = np.eye(3)
    elif(cTypeIn==CoordType.OpenGL):
      s_M_in = s_M_opgl 
    elif(cTypeIn==CoordType.Unity3D):
      s_M_in = s_M_u3d 
    elif(cTypeIn==CoordType.FreeD):
      s_M_in = s_M_fd 
    elif(cTypeIn==CoordType.UE):
      s_M_in = s_M_ue 

    # Choose correct s^M_out 
    if(cTypeOut==CoordType.SLAM):
      s_M_out = np.eye(3)
    elif(cTypeOut==CoordType.OpenGL):
      s_M_out = s_M_opgl 
    elif(cTypeOut==CoordType.Unity3D):
      s_M_out = s_M_u3d 
    elif(cTypeOut==CoordType.FreeD):
      s_M_out = s_M_fd 
    elif(cTypeOut==CoordType.UE):
      s_M_out = s_M_ue 

    # out^M_in = (s^M_out)^-1(s^M_in)
    out_M_in = (s_M_out.transpose())*s_M_in

    return out_M_in

  # main conversion funcs, better wraped in the cls
  '''
  Maths Explication:

    T_out = out^M_in * T_in

    R_out = out^M_in * R_in * in^M_out    
  '''
  @classmethod
  def convert(cls, R, t, cTypeIn=CoordType.SLAM, cTypeOut=CoordType.SLAM):

    # Identical 
    if(cTypeIn==cTypeOut):
      return R, t

    R_m = R.as_dcm()
    t_v = t.transpose() # as col vector
    o_M_i = cls.computeM_in_out(cTypeIn, cTypeOut) 

    t_v_o = o_M_i * t_v
    R_m_o = o_M_i * R_m * (o_M_i.transpose())

    R_o = Rotation.from_dcm(R_m_o) 
    t_o = t_v_o.transpose()

    return R_o, t_o
  
def unitTestConvertor():
  # init with 
  r = Rotation.from_quat([-0.5773503, -0.1924501, -0.1924501, 0.7698004])
  t = np.matrix([1,2,3])

  R_SLAM, t_SLAM = coordCvtor.from_SLAM(r,t).to_SLAM() 
  print ("R SLAM",R_SLAM.as_quat()) 
  print("\n")
  print ("t SLAM",t_SLAM)
  print("\n")

  assert((R_SLAM.as_dcm()==r.as_dcm()).all())
  assert((t_SLAM==t).all())

  # convert to U3D
  R_U3D, t_U3D = coordCvtor.from_SLAM(r,t).to_Unity3D() 
  print ("R U3D",R_U3D.as_quat())
  print("\n")
  print ("t U3D",t_U3D)
  print("\n")

  # convert to FreeD
  R_Fd, t_Fd = coordCvtor.from_Unity3D(R_U3D, t_U3D).to_FreeD() 
  print ("R FreeD",R_Fd.as_quat())
  print("\n")
  print ("t FreeD",t_Fd)
  print("\n")

  R_opgl, t_opgl = coordCvtor.from_FreeD(R_Fd, t_Fd).to_OpenGL() 
  print ("R OpenGL",R_opgl.as_quat())
  print("\n")
  print ("t OpenGL",t_opgl)
  print("\n")

  R_UE, t_UE = coordCvtor.from_OpenGL(R_opgl, t_opgl).to_UE() 
  print ("R UE",R_UE.as_quat())
  print("\n")
  print ("t UE",t_UE)
  print("\n")

  R_SLAM_, t_SLAM_ = coordCvtor.from_UE(R_UE, t_UE).to_SLAM() 
  print ("R SLAM back",R_SLAM_.as_quat())
  print("\n")
  print ("t SLAM back",t_SLAM_)
  print("\n")

  assert(np.linalg.norm(R_SLAM_.as_quat()-R_SLAM.as_quat())<10e-8)
  assert(np.linalg.norm(t_SLAM_-t_SLAM)<10e-8)

  print ("Unit test passed. \n")

if __name__ == "__main__":
  print("starting unit tests")
  unitTestConvertor()