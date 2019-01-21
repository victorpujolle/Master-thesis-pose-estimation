import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2 as cv2
import utils
import os
import time
import sys
from optparse import OptionParser
from model import Model, Output, Store
from pymanopt.manifolds import Stiefel, Rotations
from pymanopt import Problem
from pymanopt.solvers import TrustRegions
import scipy as scp
import sympy as smp
from scipy.optimize import minimize

# we use the metric system, every distance is in meter #

def readHM(filepath, name):
    '''
    read the output of the neural network and return the heatmaps
    we use plt to read the image because it is simpler than cv2
    :param filepath : path to the file
    :param name : name of the image without extention
    :param extention : extention of the image
    :param M : number of keypoints
    :return 3D array[64,64,M] with the raw keypoints
    '''

    HM = np.zeros([64,64,8])
    for i in range(8):
        name_kp = name[0:-4] + '_{:02d}'.format(i+1) + name[-4:]
        hm_name = os.path.join(filepath,name_kp)
        #print(hm_name)
        HM[:,:,i] = cv2.imread(hm_name)[:,:,0]

    return HM/255.0

def findWMax(hm):
    '''
    read the heatmap and return the coordinates of the values of the maximum of the heatmap
    :param hm : the heatmap given by readHM
    :return : [W_max, score] where W_max is a array containing the coordinates of the maximum and score is the value of the maximum
    '''
    p = hm.shape[2]
    W_max = np.zeros([2,p])
    score = np.zeros(p)

    for i in range(p):
        score[i] = np.amax(hm[:,:,i])
        (x,y) = np.where(hm[:,:,i]==score[i])
        W_max[0,i] = y[0]
        W_max[1,i] = x[0]

    return [W_max, score]

def move_vertices(vertices,R,T):
    """
    :param vertices: vertices you want to move
    :param R: rotation matrix
    :param T: translation vector
    :return: move the vertices in the space follow a rotation(R) and a translation T
    """

    rotated_vertices = R.dot(vertices)
    moved_vertices = np.copy(rotated_vertices)
    moved_vertices[0] += T[0]
    moved_vertices[1] += T[1]
    moved_vertices[2] += T[2]

    return moved_vertices

def coord2homogenouscoord(vertices):
    """
    just add a dimension with a one
    :param vertices: vertices in normal coordinate
    :return: vertices in homogenous coordinate
    """
    homo = np.ones((vertices.shape[0],vertices.shape[1]+1))
    homo[:,0:2] = vertices
    return homo

def point2vertices(points):
    """
    return the array to be ploted to plot the vertices of a cube where the order of the points is my order
    :param points: the points...
    :return: the vertices
    """
    order = [1,5,1,2,6,2,3,7,3,4,8,4,1,5,6,7,8,5]
    res = []
    for ind in order:
        res.append(points[ind-1])

    return np.array(res)

def prox_2norm(Z,lam):
    '''
    This function simplifies Z based on the value of lam and the svd of Z
    :param Z : matrix that need to be simplified
    :param lam : cutting parameter
    :return : [X, normX] this simplified matrix and the its first singular value
    '''

    [U,w,V] = np.linalg.svd(Z) # Z = U*W*V
    if np.sum(w) <= lam:
        w = [0,0]
    elif w[0] - w[1] <= lam:
       w[0] = (np.sum(w) - lam) / 2
       w[1] = w[0]
    else:
        w[0] = w[0] - lam
        w[1] = w[1]

    W = np.zeros(Z.shape)
    W[:len(Z[0]), :len(Z[0])] = np.diag(w)
    X = np.dot(U,np.dot(W,V)) # X = U*W*V
    normX = w[0]

    return X, normX

def proj_deformable_approx(X):
    '''
    Ref: A. Del Bue, J. Xavier, L. Agapito, and M. Paladini, "Bilinear
    Factorization via Augmented Lagrange Multipliers (BALM)" ECCV 2010.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; version 2, June 1991

    USAGE: Y = proj_deformable_approx(X)

    This function projects a generic matrix X of size 3*K x 2 where K is the
    number of basis shapes into the matrix Y that satisfy the manifold
    constraints. This projection is an approximation of the projector
    introduced in: M. Paladini, A. Del Bue, S. M. s, M. Dodig, J. Xavier, and
    L. Agapito, "Factorization for Non-Rigid and Articulated Structure using
    Metric Projections" CVPR 2009. Check the BALM paper, Sec 5.1.

    :param X : the 3*K x 2 affine matrix
    :return : the 3*K x 2 with manifold constraints
    '''

    r = X.shape[0]
    d = int(r / 3)
    A = np.zeros((3,3))


    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        A = A + np.dot(Ai,np.transpose(Ai))

    [U, S, V] = np.linalg.svd(A,4)

    Q = U[:,0:2]
    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = np.array([ np.trace(Ti) , Ti[1,0] - Ti[0,1] ])
        G = G + np.outer(gi,np.transpose(gi)) # it is really import to use outer and not dot !!! test yourself

    [U1, S1, V1] = np.linalg.svd(G)

    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = np.array([ Ti[0,0]-Ti[1,1] , Ti[0,1]+Ti[1,0] ])
        G = G + np.outer(gi, np.transpose(gi))

    [U2, S2, V2] = np.linalg.svd(G,4)

    if S1[0] > S2[0]:
        u = U1[:,0]
        R = [[u[0], -u[1]],[u[1], u[0]]]
    else:
        u = U2[:,0]
        R = [[u[0], u[1]], [u[1], -u[0]]]

    Q = np.dot(Q,R)

    Y = np.zeros([d*Q.shape[0],Q.shape[1]])
    L = np.zeros(d)
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        ti = 0.5*np.trace(np.dot(np.transpose(Q),Ai))
        L[i] = ti
        Y[:,2*i:2*(i+1)] = ti*Q


    return [Y, L, Q]

def syncRot(T):
    '''
    returns the rotation matrix of the approximation of the projector created by proj_deformable_approx
    :param T : the motion matrix calculated in PoseFromKpts_WP
    :return : [R,C] the rotation matrix and the values of a sorte of norm that is do not really understand
    '''
    [_, L, Q] = proj_deformable_approx(np.transpose(T))
    s = np.sign(L[0])
    C = s*L[0]
    R = np.zeros((3,3))
    R[0:2,0:3] = s*np.transpose(Q)
    R[2,0:3] = np.cross(Q[0:3,0],Q[0:3,1])

    return R,C

def estimateR_weighted(S,W,D,R0):
    '''
    estimates the update of the rotation matrix for the second part of the iterations
    :param S : shape
    :param W : heatmap
    :param D : weight of the heatmap
    :param R0 : rotation matrix
    :return: R the new rotation matrix
    '''

    A = np.transpose(S)
    B = np.transpose(W)
    X0 = R0[0:2,:]
    store_E = Store()

    [m,n] = A.shape
    p = B.shape[1]

    At = np.zeros([n, m]);
    At =  np.transpose(A)


    # we use the optimization on a Stiefel manifold because R is constrained to be othogonal
    manifold = Stiefel(n,p,1)


    ####################################################################################################################
    def cost(X):
        '''
        cost function of the manifold, the cost is trace(E'*D*E)/2 with E = A*X - B or store_E
        :param X : vector
        :return f : the cost
        '''

        if store_E.stored is None:
            store_E.stored = np.dot(A, np.transpose(X)) - B

        E = store_E.stored
        f = np.trace(np.dot(np.transpose(E),np.dot(D,E)))/2

        return f
    ####################################################################################################################

    # setup the problem structure with manifold M and cost
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)

    # setup the trust region algorithm to solve the problem
    TR = TrustRegions(maxiter=10)

    # solve the problem
    X = TR.solve(problem,X0)

    #print('X : ',X)
    return np.transpose(X) # return R = X'

def estimateC_weighted(W, R, B, D, lam):
    '''
    :param W : the heatmap
    :param R : the rotation matrix
    :param B : the base matrix
    :param D : the weight
    :param lam : lam value used to simplify some results
    :return : C0
    '''
    p = len(W[0])
    k = int(B.shape[0]/3)
    d = np.diag(D)
    D = np.zeros((2*p,2*p))
    eps = sys.float_info.epsilon

    for i in range(p):
        D[2*i, 2*i] = d[i];
        D[2*i+1, 2*i+1] = d[i];

    # next we work on the linear system y = X*C
    y = W.flatten() # vectorized W
    X = np.zeros((2*p,k)) # each colomn is a rotated Bk

    for i in range(k):
        RBi = np.dot(R,B[3*i:3*(i+1),:])
        X[:,i] = RBi.flatten()


    # we want to calculate C = pinv(X'*D*X+lam*eye(size(X,2)))*X'*D*y and then C = C'

    A = np.dot(np.dot(np.transpose(X),D),X) + lam*np.eye(X.shape[1])
    tol = max(A.shape) * np.linalg.norm(A,np.inf) * eps
    C = np.dot(np.dot(np.linalg.pinv(A),np.dot(np.transpose(X),D)),y)

    return np.transpose(C)

def PoseFromKpts_WP(W, model, D=None, verb=True, tol=1e-10):
    """
    :arg: model : Model object containing data about the shape of the bounding box
    :arg: W : the keypoints -> R(2*8) 2D keypoints
    :arg: D : the weight of each keypoints
    :return: R : rotation matrix -> R(3*3)
    :return: T : translation matrix -> T(1*3)

    the goal is to find the best rotation and tranlation matrix given the keypoints and the shape
    """

    nb_kp = 8
    k = 1
    # setting values
    if D is None:
        D = np.eye(nb_kp)
    else:
        D = np.diag(D)

    B = model.B
    mean = np.mean(B, 1)
    for i in range(3):
        B[i] -= mean[i]

    # initialization
    M = np.zeros([2, 3])
    C = 0

    # auxiliary variable for ADMM
    Z = np.copy(M)
    Y = np.copy(M)

    eps = sys.float_info.epsilon
    alpha = 1
    mu = 1 / (np.mean(W) + eps)

    # pre-computing
    BBt = np.dot(B, np.dot(D, np.transpose(B)))

    # iteration
    for iter in range(1000):

        # update translation
        T = np.sum(np.dot((W - np.matmul(Z, B)), D), 1) / (np.sum(D) + eps)  # T = sum((W-Z*B)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update motion matrix Z
        Z0 = np.copy(Z)
        Z = np.dot(np.dot(W2fit, np.dot(D, np.transpose(B))) + mu * M + Y,
                   np.linalg.inv(BBt + mu * np.eye(3 * k)))  # Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k))

        # update motion matrix M
        Q = Z - Y / mu
        X, normX = prox_2norm(np.transpose(Q), alpha / mu)
        M = np.transpose(X)
        C = normX

        # update dual variable
        Y = Y + mu * (M - Z)
        PrimRes = np.linalg.norm(M - Z) / (np.linalg.norm(Z0) + eps)
        DualRes = mu * np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0) + eps)

        # show output
        if verb:
            print('Iter = ', iter, ' ; PrimRes = ', PrimRes, '; DualRes = ', DualRes, ' ; mu = ', '{:08.6f}'.format(mu),
                  '\n')

        # check convergente
        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu = 2 * mu;
            elif DualRes > 10 * PrimRes:
                mu = mu / 2;
            else:
                pass

    # end iteration

    [R, C] = syncRot(M)
    if np.sum(np.abs(R)) == 0:
        R = np.eye(3)

    R = R[0:2, :]
    S = np.dot(np.kron(C, np.eye(3)), B)

    # iteration, part 2
    fval = np.inf

    for iter in range(1000):
        T = np.sum(np.dot((W-np.dot(R,S)),D), 1) / (np.sum(D)+eps) # T = sum((W-R*S)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update rotation
        R = np.transpose(estimateR_weighted(S, W2fit, D, R))


        # update shape
        C0 = estimateC_weighted(W2fit, R, B, D, 1e-3)[0]
        S = C0*B

        fvaltml = fval
        # fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*norm(C)^2;
        fval = 0.5*np.linalg.norm(np.dot(W2fit-np.dot(R,S),np.sqrt(D)),'fro')**2 + 0.5*np.linalg.norm(C)**2

        # show output
        if verb:
            print('Iter = ', iter, 'fval = ', fval)

        # check convergence
        if np.abs(fval-fvaltml) / (fvaltml+eps) < tol:
            break

    # end iteration
    R2 = np.zeros((3,3))
    R2[0,:] = R[0,:]
    R2[1, :] = R[1, :]
    R2[2,:] = np.cross(R[0,:],R[1, :])
    output = Output(S=S, M=M, R=R2, C=C ,C0=C0, T=T, fval=fval)

    return output

def PoseFromKpts_WP2(W, dict, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    compute the pose with weak perspective
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return ; return a Output object containing many informations
    '''

    # data size
    B = np.copy(dict.mu)  # B is the base
    pc = []
    [k,p] =  B.shape
    k = int(k/3)

    # setting values
    if weight is None:
        D = np.eye(p)
    else:
        D = np.diag(weight)

    alpha = 1

    # centralize basis
    mean = np.mean(B, 1)
    for i in range(3*k):
        B[i] -= mean[i]

    # initialization
    M = np.zeros([2, 3 * k]);
    C = np.zeros(k); # norm of each Xi

    # auxiliary variable for ADMM
    Z = np.copy(M)
    Y = np.copy(M)

    eps = sys.float_info.epsilon
    mu = 1/(np.mean(W)+eps)

    # pre-computing
    BBt = np.dot(B,np.dot(D,np.transpose(B)))

    # iteration
    for iter in range(1000):

        # update translation
        T = np.sum(np.dot((W-np.matmul(Z,B)),D), 1) / (np.sum(D)+eps) # T = sum((W-Z*B)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update motion matrix Z
        Z0 = np.copy(Z)
        Z = np.dot( np.dot(W2fit,np.dot(D,np.transpose(B))) + mu*M + Y , np.linalg.inv(BBt+mu*np.eye(3*k))) # Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k))
        # update motion matrix M
        Q = Z - Y/mu
        for i in range(k):
            [X, normX] = prox_2norm(np.transpose(Q[:,3*i:3*i+3]),alpha/mu)
            M[:, 3*i:3*i+3] = np.transpose(X)
            C[i] = normX

        # update dual variable
        Y = Y + mu*(M-Z)
        PrimRes = np.linalg.norm(M-Z) / (np.linalg.norm(Z0)+eps)
        DualRes = mu*np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0)+eps)

        # show output
        if verb:
            print('Iter = ', iter, ' ; PrimRes = ',PrimRes, '; DualRes = ', DualRes,' ; mu = ', '{:08.6f}'.format(mu), '\n')

        # check convergente
        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu = 2 * mu;
            elif DualRes > 10 * PrimRes:
                mu = mu / 2;
            else:
                pass

    # end iteration

    [R, C] = syncRot(M)

    if np.sum(np.abs(R)) == 0:
        R = np.eye(3)

    R = R[0:2,:]
    S = np.dot(np.kron(C,np.eye(3)),B)

    # iteration, part 2
    fval = np.inf

    for iter in range(1000):
        T = np.sum(np.dot((W-np.dot(R,S)),D), 1) / (np.sum(D)+eps) # T = sum((W-R*S)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update rotation
        R = np.transpose(estimateR_weighted(S, W2fit, D, R))


        # update shape
        if len(pc) == 0:
            C0 = estimateC_weighted(W2fit, R, B, D, 1e-3)[0]
            S = C0*B
        else:
            W_1 = W2fit - np.dot(np.dot(R , np.kron(C, eye(3))) , pc)
            C0 = estimateC_weighted(W_1, R, B, D, 1e-3)
            W_2 = W2fit - np.dot(np.dot(R , C0) , B)
            C = estimateC_weighted(W_2, R, pc, D, lam)
            S = np.dot(C0,B) + np.dot(np.kron(C,np.eye(3)),pc)

        fvaltml = fval
        # fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*norm(C)^2;
        fval = 0.5*np.linalg.norm(np.dot(W2fit-np.dot(R,S),np.sqrt(D)),'fro')**2 + 0.5*np.linalg.norm(C)**2

        # show output
        if verb:
            print('Iter = ', iter, 'fval = ', fval)

        # check convergence
        if np.abs(fval-fvaltml) / (fvaltml+eps) < tol:
            break

    # end iteration
    R2 = np.zeros((3,3))
    R2[0,:] = R[0,:]
    R2[1, :] = R[1, :]
    R2[2,:] = np.cross(R[0,:],R[1, :])
    output = Output(S=S, M=M, R=R2, C=C ,C0=C0, T=T, fval=fval)

    return output

def PoseFromKpts_FP(W, model, R0=None, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    compute the pose with full perspective
    solve in ||W*diag(Z)-R*S-T||^2 + ||C||^2 with S = C1*B1+...+Cn*Bn, Z denotes the depth of points
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return ; return a Output object containing many informations
    '''
    # data size
    mu = np.copy(np.transpose(model.kp_bb))
    R = np.copy(R0)
    # setting values
    if weight is None:
        D = np.eye(p)
    else:
        D = np.diag(weight)

    # centralize basis
    meanmu = np.mean(mu, 1)
    for i in range(mu.shape[0]):
        mu[i] -= meanmu[i]

    # initialization
    eps = sys.float_info.epsilon
    S = mu
    T = np.mean(W,1) * np.mean(np.std(np.dot(R[0:2,:],S),1)) / (np.mean(np.std(W,1))+eps)
    C = 0

    fval = np.inf

    # iteration
    for iter in range(1000):

        # update the depth of Z
        Z = np.dot(R,S)
        for j in range(3):
            Z[j] += T[j]

        Z = np.sum(W * Z,0) / (np.sum(W**2, 0)+eps)

        # update R and T by aligning S to W*diag(Z)
        Sp = np.dot(W, np.diag(Z))
        T = np.sum(np.dot(Sp-np.dot(R,S),D), 1) / (np.sum(np.diag(D))+eps)
        St = Sp
        for j in range(Sp.shape[0]):
            St[j] -= T[j]

        [U, _, V] = np.linalg.svd(np.dot(St,np.dot(D,np.transpose(S))))
        R = np.dot(np.dot(U,np.diag([1 , 1 , np.linalg.det(np.dot(U,V))])),V)

        fvaltml = fval
        fval = np.linalg.norm(np.dot(St-np.dot(R,S),np.sqrt(D)), 'fro')**2 + lam*np.linalg.norm(C)**2

        # show output
        if verb:
            print('Iter = ',iter, 'fval = ',fval)

        # check convergence
        if np.abs(fval-fvaltml) / (fvaltml+eps) < tol:
            break


    output = Output(S=S, R=R, C=C, T=T, Z=Z, fval=fval)

    return output

def read_text_file(name, image_nb):
    """
    :param name: name of the txt file
    :param image_nb: the index of the image
    :return name_img: name of the image
    :return joint: position of the keypoints on the image
    :return bb: position of the bounding box keypoints on the image
    :return K: intrinsec parameters of the camera (3*3 matrix)
    :return RT: extrinsec parameters of the camera (3*4 matrix)
    """

    # text file
    file = open(name, 'r')

    K  = np.zeros([3, 3])

    for i in range(image_nb-1):
        file.readline()

    line = file.readline()
    l = line.split(' ')
    del (l[-1])

    name_img = l[0]
    joints = np.reshape(np.array(list(map(int, l[1:17] ))) ,(-1,2))
    bb     = np.reshape(np.array(list(map(int, l[17:33]))), (-1,2))

    K[0, 0] = float(l[33])
    K[1, 1] = float(l[34])
    K[2 ,2] = 1.0
    K[0, 2] = float(l[35])
    K[1, 2] = float(l[36])
    K[0, 1] = float(l[37])

    RT = np.reshape(np.array([float(l[_]) for _ in range(38,50)]),(3,4))
    kps_3d = np.reshape(np.array([float(l[_]) for _ in range(50,74)]),(-1,3))

    return name_img, joints, bb, K, RT, kps_3d

def v2_read_text_file(name, image_nb):
    """
    :param name: name of the txt file
    :param image_nb: the index of the image
    :return name_img: name of the image
    :return joint: position of the keypoints on the image
    :return bb: position of the bounding box keypoints on the image
    :return K: intrinsec parameters of the camera (3*3 matrix)
    :return RT: extrinsec parameters of the camera (3*4 matrix)
    """

    # text file
    file = open(name, 'r')

    K  = np.zeros([3, 3])

    for i in range(image_nb-1):
        file.readline()

    line = file.readline()
    l = line.split(' ')
    del (l[-1])

    name_img = l[0]
    joints = np.reshape(np.array(list(map(int, l[1:17] ))) ,(-1,2))
    bb     = np.reshape(np.array(list(map(int, l[17:33]))), (-1,2))

    return name_img, joints, bb

def findRotation(S1,S2):
    '''
    find the rotation matrix between S1 and S2
    :param S1 : matrix 1
    :param S2 : matrix 2
    :return : R, the rotation R*S1 = S2
    '''
    [f,p] = S1.shape
    f = int(f/3)
    S1 = np.reshape(S1,(3,f*p))
    S2 = np.reshape(S2,(3,f*p))
    R = np.dot(S1,np.transpose(S2))
    # /!\ the matlab svd computes R = USV' and the python svd computes R = USV
    [U, _, V] = np.linalg.svd(R)
    R = np.dot(U,V)
    R = np.dot(U,  np.dot(np.diag([1.0,1.0,np.linalg.det(R)]), V))

    return R

def fullShape(S1,model):
    '''
    creates the new model besed on the S
    :param S1 : the matrix of shape
    :param model : an object of the class Model
    :return : the new object Model, and some other information and transformation matrix
    '''
    eps = sys.float_info.epsilon

    # normalization of S
    S2 = np.copy(np.transpose(model.kp_bb))
    T1 = np.mean(S1,1)
    T2 = np.mean(S2,1)
    for i in range(len(T1)):
        S1[i] -= T1[i]
        S2[i] -= T2[i]

    R = findRotation(S1,S2)
    S2 = np.dot(R,S2)
    w = np.trace(np.dot(np.transpose(S1), S2))/(np.trace(np.dot(np.transpose(S2), S2))+eps);
    T = T1 - w*np.dot(R,T2)

    vertices = np.transpose(model.vertices)

    for i in range(len(T)):
        vertices[i] = vertices[i] - T2[i]

    vertices = w*np.dot(R,vertices)

    for i in range(len(T)):
        vertices[i] = vertices[i] + T1[i]

    model_new = model.copy()
    model_new.vertices = np.transpose(vertices)
    model_new.nb_vertices = len(vertices)

    return [model_new,w,R,T]

def cost_function_v2(W, D, K, S, s=1, R=np.eye(3), T=np.array([0,0,0]),norm='fro'):
    """
    :param W: heatmap constant
    :param D: weight of the keypoints
    :param K: intrinsec camera parameters
    :param S: shape of the object
    :param s: scale variable
    :param R: rotation matrix variable
    :param T: translation variable

    :return : ||(W-P(sRS+T))*sqrt(D)||^2 and is useable by pymanopt
    """

    # P : extrasec camera parameters in the camera referenciel : P = K*[id|0]
    P = np.concatenate((K, np.array([[0, 0, 0]]).T), axis=1)

    T_mat = np.zeros((3, S.shape[0]))
    for i in range(S.shape[0]):
        T_mat[:, i] = T

    # X = P.(sRS + T)
    X = np.dot(P, (np.concatenate((s * np.dot(R, S.T) + T_mat, [np.ones((8))]), axis=0)))

    # normalization of the colomn of X by the last value
    X = X / X[2]

    # X = (W-P(sRS+T))*D^1/2
    X = (W - X[0:2]) * np.sqrt(D[:])

    cost = np.linalg.norm(X)
    return cost

def optimizer_R_v1(W, D, S, R0):
    """
    :param W: heatmap constant
    :param D: weight of the keypoints
    :param S: shape of the object
    :param R0: initial R

    :return : the optimal R with fixed T and s
    the cost is ||(W-(RS))*sqrt(D)|| but in fact is the scale factor is already taken into account in S and T is taken into account in W
    """

    # this store object is needed because the manifold optimizer do not works if it is not implemented like that
    store = Store()

    # -------------------------------------COST FUNCTION-------------------------------------
    def cost(R):
        """
        :param R: rotation matrix variable
        :return : ||(W-(RS))*D^1/2||^2 = tr((W-(RS))*D*(W-(RS))')
        """

        if store.stored is None:
            store.stored = W - np.dot(R,S)

        X = store.stored
        f = np.trace(np.dot(X, np.dot(D, np.transpose(X)))) / 2

        return f
    # ----------------------------------------------------------------------------------------

    # we use the optimization on a Stiefel manifold because R is constrained to be othogonal
    manifold = Stiefel(3, 2, 1)

    # setup the problem structure with manifold M and cost
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)

    # setup the trust region algorithm to solve the problem
    TR = TrustRegions(maxiter=15)

    # solve the problem
    R_opt = TR.solve(problem, R0)

    return np.transpose(R_opt)

def optimizer_s_v1(W, D, S, R):
    """
    :param W: heatmap constant
    :param D: weight of the keypoints
    :param S: shape of the object
    :param R: rotation matrix

    :return : the optimal s with fixed R
    """

    eps = sys.float_info.epsilon
    lam = 1e-3

    # next we work on the linear system y = X*s => X = y/s (s is a scalar) 1/s = X*D*Xt
    y = W.flatten()  # vectorized W
    RS = np.dot(R,S)
    X = RS.flatten() # RS but flattened

    # Solving the problem y = X*s this means s = 1/(X'*D*X+lam*eye(size(X,2))) * X'*D * y
    a = np.dot(np.dot(X, D), X.T) + lam
    s = np.dot(X.T.dot(D)/a, y)

    return s

def weak_model_opt(W, D, model, lam=1, tol=1e-10, verb=True):

    """
    :param W: detection keypoints
    :param D: weight of each keypoints
    :param model: shape model of the object
    :param lam:
    :param tol: cut value
    :param verb: verbosity
    :return: s,R,T : scale rotation and translation

    I do not understand how this algo works, it is numerical optimization from the paper -6-DoF Object Pose from Semantic Keypoints-
    """

    # B is the normalized shape matrix. The lines (x,y,z coord) have a null average and a sd of 1
    # B will be used as the base, it is centralized and normalized
    B = np.copy(model.B)

    # D become a diag matrix
    D = np.diag(D)

    # number of kp
    nb_kp = B.shape[1]

    # variable initialization
    M = np.zeros([2,3])

    # variables initialization for ADMM (alternating direction method of multipliers)
    Z = np.zeros([2,3])
    Y = np.zeros([2,3])

    eps = sys.float_info.epsilon # smallest value of the computer
    mu = 1/(np.mean(W)+eps)

    # pre computing of B*D*Bt
    BBt = np.dot(B,np.dot(D,B.T))


    # ------------------------------------- ITERATION -------------------------------------
    for iter in range(1000):
        # update translation
        # T is the 2D translation
        T = np.sum(np.dot(W - np.matmul(Z, B), D), 1) / (np.sum(D) + eps)

        # alignement of the detected keypoints
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update motion matrix Z
        Z0 = np.copy(Z)
        Z = np.dot(np.dot(W2fit, np.dot(D, B.T)) + mu * M + Y, np.linalg.inv(BBt + mu * np.eye(3)))  # Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k))

        # update motion matrix M
        Q = Z - Y / mu

        [X, normX] = prox_2norm(Q.T,1/mu)
        M = X.T
        s = normX

        # update dual variable
        Y = Y + mu * (M - Z)
        PrimRes = np.linalg.norm(M - Z) / (np.linalg.norm(Z0) + eps)
        DualRes = mu * np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0) + eps)

        # show output
        if verb: print('Iter = ', iter, ' ; PrimRes = ', '{:08.12f}'.format(PrimRes), '; DualRes = ', '{:08.12f}'.format(DualRes), ' ; mu = ', '{:08.6f}'.format(mu))

        # check convergente
        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu = 2 * mu;
            elif DualRes > 10 * PrimRes:
                mu = mu / 2;
            else:
                pass
    # ----------------------------------- END ITERATION -----------------------------------

    # computation of the Rotation matrix and the scale from the projector M
    R, s = syncRot(M)

    if np.sum(np.abs(R)) == 0:
        R = np.eye(3)
    R = R[0:2, :] # reduction of R because we are in a weak perpective model

    # S is the base, but rescaled by the factor of scale s
    S = s*B

    s0 = s

    # cost initialization
    fval = np.inf

    # for the optimization of s it is iteresting to compute D, but twice as big
    D2 = np.zeros((2 * nb_kp, 2 * nb_kp))  # the weight matrix, but twice as big
    for i in range(nb_kp):
        D2[2 * i, 2 * i] = D[i,i];
        D2[2 * i + 1, 2 * i + 1] = D[i,i];

    # ------------------------------------- ITERATION -------------------------------------
    for iter in range(1000):
        # update translation
        # T is the 2D translation
        T = np.sum(np.dot((W-np.dot(R,S)),D), 1) / (np.sum(D)+eps) # T = sum((W-R*S)*D, 1) / (sum(D)+eps)

        # alignement of the detected keypoints
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update rotation
        R = optimizer_R_v1(W2fit, D, B, R).T

        # update scale
        s0 = optimizer_s_v1(W2fit,D2,B,R)
        S = s0 * B

        fvaltml = fval
        # fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*abs(s)**2
        fval = 0.5 * np.linalg.norm(np.dot(W2fit - np.dot(R, S), np.sqrt(D)), 'fro') ** 2 + 0.5*abs(s)**2

        # show output
        if verb:
            print('Iter = ', iter, 'fval = ', '{:08.12f}'.format(fval))

        # check convergence
        if np.abs(fval - fvaltml) / (fvaltml + eps) < tol: break

    R2 = np.zeros((3, 3))
    R2[0, :] = R[0, :]
    R2[1, :] = R[1, :]
    R2[2, :] = np.cross(R[0, :], R[1, :])

    return s,s0, R2, T

def full_model_opt(W, D, model, s0=1, R0=np.eye(3), lam=1, tol=1e-10, verb=True):
    """
    :param W: detection keypoints
    :param D: weight of each keypoints
    :param model: shape model of the object
    :param s0: initial scale
    :param R0: initial rotation matrix
    :param T0: initial translation
    :return: s,R,T : scale rotation and translation

    I do not understand how this algo works, it is numerical optimization from the paper -6-DoF Object Pose from Semantic Keypoints-
    """

    # B is the normalized shape matrix. The lines (x,y,z coord) have a null average and a sd of 1
    # B will be used as the base, it is centralized and normalized
    B = np.copy(model.B)

    # R is the rotation matrix
    R = np.copy(R0)

    # D become a diag matrix
    D = np.diag(D)

    # number of kp
    nb_kp = B.shape[1]

    eps = sys.float_info.epsilon  # smallest value of the computer

    # initialisation
    T = np.mean(W, 1) * np.mean(np.std(np.dot(R[0:2, :], B), 1)) / (np.mean(np.std(W, 1)) + eps)
    s = 0

    fval = np.inf

    # iteration
    for iter in range(10000):

        # update the depth of Z
        Z = np.dot(R, B) # base rotation

        for j in range(3): # base translation
            Z[j] += T[j]

        Z = np.sum(W * Z, 0) / (np.sum(W ** 2, 0) + eps) # application of the depth over the keypoints and normalization

        # update R and T by aligning B to W*diag(Z)
        Sp = np.dot(W, np.diag(Z))
        T = np.sum(np.dot(Sp - np.dot(R, B), D), 1) / (np.sum(np.diag(D)) + eps)
        for j in range(Sp.shape[0]):
            Sp[j] -= T[j]

        # update R with Procruste orthogonal
        [U, _, V] = np.linalg.svd(np.dot(Sp, np.dot(D, B.T)))
        R = np.dot(np.dot(U, np.diag([1, 1, np.sign(np.linalg.det(np.dot(U, V)))])), V)

        # update s by least square
        RB = R.dot(B) # R.B
        WZ = Sp # W.Z (centralized = takes T in account)

        sum1 = 0 # sigma(ij)[D(i)(WZ)(ij)(RB)(ij)]
        sum2 = 0 # sigma(ij)[(RB)(ij)^2]

        for i in range(nb_kp):
            for j in range(3):
                sum1 += D[i,i]*WZ[j,i]*RB[j,i]
                sum2 += RB[j,i]**2

        #break
        s = sum1/(sum2)
        B = s*B # scaling the base

        fvaltml = fval
        fval = np.linalg.norm(np.dot(Sp - np.dot(R, B), np.sqrt(D)), 'fro') ** 2

        # show output
        if verb:
            print('Iter = ', iter, 'fval = ', fval, 's = ',s)

        # check convergence
        if np.abs(fval - fvaltml) / (fvaltml + eps) < tol:
            break

    return s, Z, R, T

def shape_projection_wp(s, R, T, model, box=False):
    """
    :param s: scale
    :param R: rotation matrix
    :param T: translation vector
    :param model: shape model (the one that was used in weak_model_opt)
    :param point: if True model.B is used, if False model.B is converted into the vertcies of the bb (use this to plot the bb)
    :return: the new shape S
    """

    # rescaling
    if box:
        S = s * point2vertices(model.B.T).T
    else:
        S = s * model.B

    # rotation
    S_new = np.dot(R, S)

    # translation
    S_new[0] += T[0]
    S_new[1] += T[1]

    eps = sys.float_info.epsilon

    return S_new

def anglerotation(R):
    """
    compute the angle of the rotation matrix R
    :param R: the rotation matrix
    :return: the angle of rotation in degree
    """
    theta = np.rad2deg(np.arccos((np.trace(R)-1)/2))
    return theta

def visualrotation(R1,R2):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    repere = np.array([[0,0,0],[0,0,1],[0,0,0],[0,1,0],[0,0,0],[1,0,0]]).T

    zline = repere[0]
    xline = repere[1]
    yline = repere[2]
    ax.plot3D(xline, yline, zline, 'gray')

    Rot1 = np.dot(R1,repere)
    Rot2 = np.dot(R2,repere)

    zline = Rot1[0]
    xline = Rot1[1]
    yline = Rot1[2]
    ax.plot3D(xline, yline, zline, 'red')

    zline = Rot2[0]
    xline = Rot2[1]
    yline = Rot2[2]
    ax.plot3D(xline, yline, zline, 'blue')

    plt.show()






