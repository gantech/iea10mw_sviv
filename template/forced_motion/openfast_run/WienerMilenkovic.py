import numpy as np

def applyWM(c,v,transpose=1):
    """Apply a rotation defined by Wiener-Milenkovic parameters in 'c' to a vector 'v'. Optionally transpose the rotation """

    magC = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
    c0 = 2.0-0.125*magC
    nu = 2.0/(4.0-c0)
    cosPhiO2 = 0.5*c0*nu
    cCrossV = np.cross(c,v)
    return v + transpose * nu * cosPhiO2 * cCrossV + 0.5 * nu * nu * np.cross(c,cCrossV)

def calcH(c):
    """Calculate H matrix for a given Wiener-Milenkovic parameter 'c'

       Implementation copied from https://github.com/OpenFAST/openfast/blob/main/modules/beamdyn/src/BeamDyn_Subs.f90#L175

       Supposed to be Eq. 35 in Wang et al. "BeamDyn: A hig-fidelity wind turbine blade solver in the FAST modular framework",
       Wind Energy, 2016.

    """

    cf = c * 0.25
    cq = np.dot(c,c)
    aa = 2.0 * (1.0 + cq) * (1.0 + cq)
    cb0 = 2.0 * (1.0 - cq) / aa
    cb = c / aa

    Hh = np.zeros(3,3)
    Hh[0,0] = cb[0] * cf[0] + cb0
    Hh[1,0] = cb[1] * cf[0] + cb[2]
    Hh[2,0] = cb[2] * cf[0] - cb[1]

    Hh[0,1] = cb[0] * cf[1] - cb[2]
    Hh[1,1] = cb[1] * cf[1] + cb0
    Hh[2,1] = cb[2] * cf[1] + cb[0]

    Hh[0,2] = cb[0] * cf[2] + cb[1]
    Hh[1,2] = cb[1] * cf[2] - cb[0]
    Hh[2,2] = cb[2] * cf[2] + cb0

    return Hh


def compose(c1, c2, transC1=1.0):
    """Compose two vectors of Wiener-Milenkovic parameters. Optionally transpose the first rotation c1."""

    c1Plusc2 = np.zeros(np.shape(c1))

#    if(transC1):
#        print("Transposing c1")
#        c1 = -c1
    c10 = 2.0 - 0.125*np.dot(c1,c1)
    c20 = 2.0 - 0.125*np.dot(c2,c2)
    delta1 = (4.0-c10)*(4.0-c20)
    delta2 = c10*c20 - transC1 * np.dot(c1,c2)

    if (delta2 < 0):
        premult_fac = -4.0 / (delta1 - delta2)
    else:
        premult_fac = 4.0 / (delta1 + delta2)

    c1Plusc2 = premult_fac * (c10 * c2 + c20 * transC1 * c1 + transC1 * np.cross(c1,c2))

#     if(delta2 < 0):
# #        print("delta2 is < 0")
#         c1Plusc2 = -4.0*(c20*c1 + c10*c2 + np.cross(c1,c2))/(delta1-delta2)
#     else:
# #        print("delta2 is > 0")
#         c1Plusc2 = 4.0*(c20*c1 + c10*c2 + np.cross(c1,c2))/(delta1+delta2)

    return c1Plusc2

def wmToR(c):
    """Converts a vector of Wiener-Milenkovic parameters into a rotation tensor"""
    magC = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
    c0 = 2.0-0.125*magC
    nu = 2.0/(4.0-c0)
    cosPhiO2 = 0.5*c0*nu
    ss_c = skewSym(c)
    R = np.identity(3) + nu * cosPhiO2 * ss_c + 0.5*nu*nu*np.matmul(ss_c,ss_c)
    return R

def phiNtoR(phi,n):
    ss_n = skewSym(n)
    return np.identity(3) + np.sin(phi) * ss_n + (1.0 - np.cos(phi)) * np.matmul(ss_n,ss_n)

def skewSym(n):
    """Returns a 3x3 skew-symmetric tensor from a 3x1 vector"""
    ss = np.zeros((3,3))
    ss[0,1] = -n[2]
    ss[0,2] = n[1]
    ss[1,0] = n[2]
    ss[1,2] = -n[0]
    ss[2,0] = -n[1]
    ss[2,1] = n[0]
    return ss

def test_time():
    import timeit
    wmTime = timeit.timeit('wmToR([0.45936844,  0.45936844,  0.45936844])', number=100000, globals=globals())
    phiNTime = timeit.timeit('phiNtoR(0.7853981633974483, [0.57735027,  0.57735027,  0.57735027])', number=100000, globals=globals())
    print("Execution time for Wiener-Milenkovic = {}".format(wmTime))
    print("Execution time for Phi N = {}".format(phiNTime))
    print("Wiener-Milenkovic implementation is {:.1f}% faster than Phi N implementation".format( (phiNTime-wmTime)/phiNTime*100 ))

def test_composeMinus():

    wm_rot1 = 4.0*np.tan(0.25* np.radians(45.0)) * np.array([0,0,1])
    wm_rot2 = 4.0*np.tan(0.25* np.radians(45.0)) * np.array([0,1,0])

    print(wm_rot1)
    print(compose(np.zeros(3), wm_rot1 ))
    print(compose(wm_rot2, compose(wm_rot2, wm_rot1), -1))

    print(applyWM(wm_rot2, applyWM(wm_rot1, np.array([1,1,1]))) )
    print(applyWM(compose(wm_rot2, wm_rot1), np.array([1,1,1])) )

def interp_rot(q1, q2, interp_fac):
    """Interpolate between two Wiener-Milenkovic parameters q1 and q2 with
    interpolation factor interp_fac

    Args:
        q1 (np.array): Starting Wiener-Milenkovic parameter
        q2 (np.array): End Wiener-Milenkovic parameter
        interp_fac (double): Interpolation factor

    Return:
        q_interp (np.array): Interpolated Wiener-Milenkovic parameter

    """

    return compose( interp_fac * compose(q1, q2, -1), q1)


def test_interp_rot():

    q1 = 4.0*np.tan(0.25* np.radians(0.0)) * np.array([1,0,0])
    q2 = 4.0*np.tan(0.25* np.radians(45.0)) * np.array([1,0,0])

    qinterp_gold = 4.0*np.tan(0.25* np.radians(0.5*45.0)) * np.array([1,0,0])
    np.set_printoptions(precision=16)
    print(interp_rot(q1,q2,0.5))
    print(applyWM(qinterp_gold, np.array([3,3,3]) ))
    print(qinterp_gold - interp_rot(q1,q2,0.5))
    print(applyWM(interp_rot(q1,q2,0.5), np.array([3,3,3])))



def test_pitch_conetilt():

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    rloc = 45.0

    time = 0.0
    omega = (12.1/60.0)*2.0*np.pi
    theta = omega * time
    sinOmegaT = np.sin(theta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #Final orientation is a composition of the following in order
    #Blade (0,1,2), Cone, Tilt, Hub rotation, Pitch, Local deformation

    #Test with zeros
    #tilt = np.radians(0.0)
    #cone = np.radians(0.0)
    #pitch = np.radians(0.0)
    #loc_deform = np.radians(0.0)

    tilt = np.radians(0.0)
    cone = np.radians(-2.5)
    pitch = np.radians(0.0)
    loc_deform = np.radians(0.0)

    wm_tilt = 4.0*np.tan(0.25 * tilt) * np.array([0.0,1.0,0.0])
    tilt_axis = applyWM( wm_tilt, np.array([1.0,0.0,0.0]) )

    theta = np.radians(90)
    wm_hubrot = 4.0*np.tan(0.25*theta) * tilt_axis

    print( applyWM(4.0*np.tan(0.25 * np.radians(90.0)) * np.array([1,0,0]), [0,1,0]) )

    print(np.array([0,1,0]) + applyWM(wm_hubrot, [0,0,60]) + applyWM(4.0*np.tan(0.25 * np.radians(90.0)) * np.array([1,0,0]), [0,1,0]) - np.array([0,1,0]) )

    for iBlade in [0,1,2]:

        wm_rotblade_ref = np.array([4.0 * np.tan(0.25 * np.radians(iBlade * 120.0)), 0.0, 0.0 ])
        cone_axis = applyWM(wm_rotblade_ref, [0,1,0])
        wm_cone = 4.0 * np.tan(0.25 * cone) * cone_axis

        #The order of composing here is important. Most other orders won't work.
        #The first rotation needs to come last
        wm_rotblade = compose(wm_hubrot, compose(wm_tilt, compose(wm_cone, wm_rotblade_ref)))

        pitch_axis = applyWM(wm_rotblade, np.array([0,0,1]))
        wm_pitch = 4.0 * np.tan(0.25 * pitch) * pitch_axis

        wm_blade_pitch = compose(wm_pitch, wm_rotblade)

        #Deformation of 45 degrees about 1/sqrt(3)[1,1,1] in the local frame of reference
        wm_loc_deform = 4.0 * np.tan(0.25 * loc_deform) * applyWM(wm_blade_pitch, np.ones(3) / np.sqrt(3.0))
        wm_final_orient = compose(wm_loc_deform, wm_blade_pitch)

        bldPt = applyWM(wm_blade_pitch, [0,0,rloc])
        ax.quiver(0,0,0,bldPt[0], bldPt[1], bldPt[2], normalize=True,length=rloc)

        print(iBlade, ' ', wm_final_orient)

        xp = applyWM(wm_final_orient, [1,0,0])
        yp = applyWM(wm_final_orient, [0,1,0])
        zp = applyWM(wm_final_orient, [0,0,1])

        ax.quiver(bldPt[0], bldPt[1], bldPt[2], xp[0], xp[1], xp[2], length=30.0)
        ax.quiver(bldPt[0], bldPt[1], bldPt[2], yp[0], yp[1], yp[2], length=30.0, linestyle='--')
        ax.quiver(bldPt[0], bldPt[1], bldPt[2], zp[0], zp[1], zp[2], length=30.0, linestyle=':')

        xp = applyWM(wm_rotblade, [1,0,0])
        yp = applyWM(wm_rotblade, [0,1,0])
        zp = applyWM(wm_rotblade, [0,0,1])

        ax.quiver(bldPt[0], bldPt[1], bldPt[2], xp[0], xp[1], xp[2], length=30.0)
        ax.quiver(bldPt[0], bldPt[1], bldPt[2], yp[0], yp[1], yp[2], length=30.0, linestyle='--')
        ax.quiver(bldPt[0], bldPt[1], bldPt[2], zp[0], zp[1], zp[2], length=30.0, linestyle=':')


    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(-100,100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    plt.close(fig)



if __name__=="__main__":

    #test_pitch_conetilt()

    #test_composeMinus()

    test_interp_rot()
