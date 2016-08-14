import numpy as np
import lxml.etree as ET
from models.gpr import GaussianProcessModel 

class GaussianProcessParser():

    def __init__(self):
        """Create a GaussianProcess Parser that can parse PMML files"""
        self.nsp = "{http://www.dmg.org/PMML-4_3}";


    def parse(self,filename):
        """Parse a Gaussian Process PMML file. Return a GaussianProcessModel"""
        GPM = self._parse_GPM(filename)
        featureName,targetName = self._parse_name(GPM)
        kernelName,k_lambda,nugget,gamma = self._parse_kernel(GPM)
        xTrain,yTrain = self._parse_training_values(GPM)
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        return GaussianProcessModel(gamma=gamma,nugget=nugget,k_lambda=k_lambda,
            kernelName=kernelName,xTrain=xTrain,yTrain=yTrain)


    def _parse_GPM(self,filename):
        """Return the PMML document as an etree element"""
        tree = ET.parse(filename)
        root = tree.getroot()
        tagname = "GaussianProcessModel".lower()
        GPM = root.find(self.nsp + tagname)
        if GPM is None:
            raise "Missing tag %s"%tagname
        return GPM


    def _parse_name(self,GPM):
        """parse MiningSchema for features and targets"""
        # Will get a list of target name and feature name
        tagname = "MiningSchema".lower();
        MS=GPM.find(self.nsp+tagname)
        targetName=[]
        featureName=[]
        for MF in MS:
            MF_name=MF.attrib["name"]
            MF_type=MF.attrib["usagetype"]
            if MF_type == "active":
                featureName.append(MF_name)
            elif MF_type == "predicted":
                targetName.append(MF_name)

        return featureName,targetName


    def _parse_kernel(self,GPM):
        """Return kernel parameters"""
        kernelName = None;

        name = "ARDSquaredExponentialKernel".lower()
        kernel = GPM.find(self.nsp+name)
        if kernel is not None:
            kernelName = "ARDSquaredExponentialKernelType"
            nugget = float(kernel.attrib["noisevariance"])
            gamma = float(kernel.attrib["gamma"])
            array = kernel.find(self.nsp+"lambda").find(self.nsp+"array").text
            array = array.strip()
            k_lambda = [float(i) for i in array.split(" ")]


        name = "AbsoluteExponentialKernelType".lower()
        kernel = GPM.find(self.nsp+name)
        if kernel is not None:
            kernelName = "AbsoluteExponentialKernelType"
            array = kernel.find(self.nsp+"lambda").find(self.nsp+"array").text
            array = array.strip()
            k_lambda = [float(i) for i in array.split(" ")]
            nugget = float(kernel.attrib["noisevariance"])
            gamma = float(kernel.attrib["gamma"])

        if kernelName is None:
            raise "Unable to find valid kernel tag"

        return kernelName,k_lambda,nugget,gamma


    def _parse_training_values(self,GPM):
        """Return the training values"""
        traininginstances = GPM.find(self.nsp+"traininginstances")
        inlinetable = traininginstances.find(self.nsp+"inlinetable")
        instancefields = traininginstances.find(self.nsp+"instancefields")

        [features,targets] = parse_name(self.nsp,GPM)

        nrows = int(traininginstances.attrib['recordcount'])
        fcols = len(features)
        tcols = len(targets)

        xTrain = np.zeros([nrows,fcols]);
        yTrain = np.zeros([nrows,tcols]);

        for i,row in enumerate(inlinetable.findall(self.nsp+"row")):
            for j,featureName in enumerate(features):
                col = row.find(self.nsp+featureName)
                xTrain[i][j] = float(col.text)

            for j,featureName in enumerate(targets):
                col = row.find(self.nsp+featureName)
                yTrain[i][j] = float(col.text)

        return xTrain,yTrain


