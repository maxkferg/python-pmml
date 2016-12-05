import numpy as np
import lxml.etree as ET
from numpy import sqrt,exp
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
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.k_lambda = k_lambda
        self.beta = beta
        self.gamma = gamma
        self.nugget = nugget
        self.kernelName = kernelName
        
        # Setup the regressor as if gp.fit had been called
        # See https://github.com/scikit-learn/scikit-learn/master/sklearn/gaussian_process/gpr.py
        kernel = self._getKernel()
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gp.K = kernel(xTrain);
        gp.X_train_ = xTrain;
        gp.y_train_ = yTrain;
        gp.L_ = cholesky(gp.K, lower=True)
        gp.alpha_ = cho_solve((gp.L_, True), yTrain)
        gp.fit(xTrain,yTrain)
        gp.kernel_ = kernel
        self.gp = gp
        self.kernel = kernel

        # Calculate the matrix inverses once. Save time later
        # This is only used for own own implimentation of the scoring engine
        self.L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)

    def score(self,xnew):
        """
        Generate scores for new x values
        xNew should be an array-like object where each row represents a test point
        Return the predicted mean and standard deviation [mu,s]
        @param{np.Array} xnew. An numpy array where each row corrosponds to an observation
        @output{Array} mu. A list containing predicted mean values
        @output{Array} s. A list containing predicted standard deviations
        """
        self._validate_xnew(xnew)
        #mu,sd = self.gp.predict(xnew,return_std=True)
        #return {'mu':mu.T.tolist()[0], 'sd':sd.tolist()}

        #K_trans = self.kernel(X, self.xTrain)
        #y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        #y_mean = self.y_train_mean + y_mean  # undo normal.


        # Compute variance of predictive distribution
        #y_var = self.kernel_.diag(X)
        #y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        #y_var_negative = y_var < 0
        #if np.any(y_var_negative):
        #    warnings.warn("Predicted variances smaller than 0. "
        #                  "Setting those variances to 0.")
        #    y_var[y_var_negative] = 0.0
        #return y_mean, np.sqrt(y_var)


    def valid(self):
        """Check that all of the parameters are valid. Throw error on failure"""
        pass


    def _validate_xnew(self,xnew):
        """Ensure that the size of xnew matches the expected length"""
        rows,cols = xnew.shape
        nparams = len(self.k_lambda)
        if nparams != cols:
            raise ValueError("Expected %i elements in xNew, not %i"%(nparams,cols))



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
        elif self.kernelName=="ARDSquaredExponentialKernel":
            return self.gamma * RBF(self.k_lambda) + WhiteKernel(noise_level=self.nugget)
        elif self.kernelName=="AbsoluteExponentialKernel":
            raise Exception('AbsoluteExponentialKernel not implimented yet')
        elif self.kernelName=="GeneralizedExponentialKernel":
            raise Exception("GeneralizedExponentialKernel not implimented yet")
        else:
            raise Exception("Unknown kernel "+self.kernelName)













