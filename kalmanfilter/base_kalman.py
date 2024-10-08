# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

"""
    Base Kalman Filter for MOT
"""

import numpy as np

from abc import ABCMeta, abstractmethod


class KF_Base(object):
    __metaclass__ = ABCMeta

    def __init__(self, n, m, P=None, Q=None, R=None, init_x=None, cfg=None):
        """
        Creates a KF object with n states, m observables, and specified values for
        prediction noise covariance pval, process noise covariance qval, and
        measurement noise covariance rval.
        shape: P-->NxN,Q-->NxN,R-->MxM
        """
        # No previous prediction noise covariance

        self.n = n
        self.m = m
        self.Q = Q
        self.R = R
        self.cfg = cfg

        # Current state is zero, with diagonal noise covariance matrix
        if init_x is not None:
            self.x = np.array(init_x).reshape(n, 1)
        else:
            self.x = np.zeros((n, 1))

        self.P = P

        # Get state transition and measurement Jacobians from implementing class
        #         self.F = self.getF(self.x)
        #         self.H = self.getH(self.x)

        # Set up covariance matrices for process noise and measurement noise
        # self.Q = self.getQ()  # np.eye(n) * qval
        # self.R = self.getR()  # np.eye(m) * rval

        # Identity matrix will be usefel later
        self.I = np.eye(n)

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

    def predict(self):
        """
        Predict next state in dt time
        """
        return np.array(self.f(self.x).reshape(self.n))

    @abstractmethod
    def f(self, x):
        """
        Your implementing class should define this method for the state transition function f(x),
        returning a NumPy array of n elements.  Typically this is just the identity function np.copy(x).
        """
        raise NotImplementedError()

    @abstractmethod
    def getF(self, x):
        """
        Your implementing class should define this method for returning the n x n Jacobian matrix F of the
        state transition function as a NumPy array.  Typically this is just the identity matrix np.eye(n).
        """
        raise NotImplementedError()

    @abstractmethod
    def h(self, x):
        """
        Your implementing class should define this method for the observation function h(x), returning
        a NumPy array of m elements. For example, your function might include a component that
        turns barometric pressure into altitude in meters.
        """
        raise NotImplementedError()

    @abstractmethod
    def getH(self, x):
        """
        Your implementing class should define this method for returning the m x n Jacobian matirx H of the
        observation function as a NumPy array.
        """
        raise NotImplementedError()
