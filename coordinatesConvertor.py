import numpy as np
import math
from pyquaternion import Quaternion

# Convert world coordiantes to relative coordiantes
# ALL the inputs are reprensented in the WORLD FRAME

'''
T0: PREVIOUS translation vector: 
T0 = [t0, t1, t2]  
T1: CURRENT translation vector:  
T1 = [t0, t1, t2]  

Q0: PREVIOUS quaternion vector:  
Q0 = Quaternion[x=q0, y=q1, z=q2, w=q3]
Q1: CURRENT quaternion vector:
Q1 = Quaternion[x=q0, y=q1, z=q2, w=q3]

return:
rT1, rq1: RELATIVE pose by taking t0 and Q0 as origin
'''

def world2Relative(t0, t1, Q0, Q1):

    assert(np.shape(t0)==np.shape(t1)==(3,))

    # Exploit the pyquaternion for representation of quaternion
    q0 = Quaternion(Q0)
    q1 = Quaternion(Q1)

    # Get rotation matrix
    r0 = q0.rotation_matrix
    r1 = q1.rotation_matrix

    # Calculate rQ1
    rq1 = q1*Quaternion(q0.inverse)


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

    # Exploit the pyquaternion for representation of quaternion
    q0 = Quaternion(Q0)
    q1 = Quaternion(Q1)

    # Get rotation matrix
    r1 = q1.rotation_matrix

    # Calcualte rQ1
    wq1 = q1*q0

    # Calculate rT1
    wT1 = np.dot(r1,t0)+t1

    return wq1, wT1

