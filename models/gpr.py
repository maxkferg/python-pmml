import numpy as np
import lxml.etree as ET
from numpy import sqrt,exp
from helpers import parser,scorer,translator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.linalg import cholesky, cho_solve, solve_triangular

class GaussianProcessModel:
    """
    Represents a trained Gaussian Process Regression Model.
    Can be initialized from a scikit GaussianProcessRegressor or a PMML file
    """
    def __init__(self,gamma,beta,nugget,kernelName,k_lambda,xTrain,yTrain):
        """
        Create a new GaussianProcess Object
        gamma: Hyperparameter
        beta: Hyperparameter
        k_lambda: Hyperparameter
        nugget: The noise hyperparameter
        kernelName: The name of the covariance kernel
        xTrain: Numpy array containing x training values
        yTrain: Numpy array containing y training values

        """
        self.xTrain=xTrain
        self.yTrain=yTrain
        self.k_lambda=k_lambda
        self.beta=beta
        self.gamma=gamma
        self.nugget=nugget
        self.kernelName=kernelName


    def score(self,xNew):
        """
        Generate scores for new x values
        xNew should be an array-like object where each row represents a test point
        Return the predicted mean and standard deviation [mu,s]
        @param{Array} xNew. An array of x values where each row corrosponds to a point
        @output{Array} mu. A column vector containing predicted mean values
        @output{Array} s. A column vector containign predicted standard deviations
        """
        kernel = self._getKernel()
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
        # Setup the regressor as if gp.fit had been called
        # See https://github.com/scikit-learn/scikit-learn/master/sklearn/gaussian_process/gpr.py
        gp.K = kernel(self.xTrain);
        gp.X_train_ = self.xTrain;
        gp.y_train_ = self.yTrain;
        gp.L_ = cholesky(gp.K, lower=True)
        gp.alpha_ = cho_solve((gp.L_, True), self.yTrain)
        gp.fit(self.xTrain,self.yTrain)
        gp.kernel_ = kernel;
        return gp.predict(xNew,return_std=True)

    def valid(self):
        """Check that all of the parameters are valid. Throw error on failure"""
        pass

    def _toPMML(self,filename):
        """Write the trained model to PMML. Return PMML as string"""
        X = self.xTrain;
        Y = self.yTrain;
        gamma = self.gamma
        nugget = self.nugget
        k_lambda = self.k_lambda
        copywrite = "DMG.org"
        xrow,yrow,xcol,ycol = translator.trans_get_dimension(X,Y)
        featureName,targetName = translator.trans_name(xcol, ycol)
        # Start constructing the XML Tree
        PMML = translator.trans_root(None,copywrite,None)
        PMML = translator.trans_dataDictionary(PMML,featureName,targetName,xcol,ycol)
        GPM = translator.trans_GP(PMML)
        GPM = translator.trans_miningSchema(GPM,featureName,targetName)
        GPM = translator.trans_output(GPM)
        GPM = translator.trans_kernel(GPM,k_lambda,nugget,gamma,xcol,'squared_exponential')
        GPData = translator.trans_traininginstances(GPM,xrow,xcol+ycol)
        translator.trans_instancefields(GPData,featureName,targetName)
        translator.trans_inlinetable(GPData,featureName,targetName,X,Y)
        # Write the tree to file
        tree = ET.ElementTree(PMML)
        tree.write(filename,pretty_print=True,xml_declaration=True,encoding="utf-8")
        print 'Wrote PMML file to %s'%filename


    def _getKernel(self):
        """Get the right kernel according to the kernelName parameter"""
        if self.kernelName=="RadialBasisKernel":
            raise Exception('RadialBasisKernel not implimented yet')
        elif self.kernelName=="ARDSquaredExponentialKernelType":
            return self.gamma * RBF(self.k_lambda) + WhiteKernel(noise_level=self.nugget)
        elif self.kernelName=="AbsoluteExponentialKernel":
            raise Exception('AbsoluteExponentialKernel not implimented yet')
        elif self.kernelName=="GeneralizedExponentialKernel":
            raise Exception("GeneralizedExponentialKernel not implimented yet")
        else:
            raise Exception("Unknown kernel "+self.kernelName)













