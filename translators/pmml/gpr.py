# This file is part of the PMML package for python.
#
# The PMML package is free software: you can redistribute it and/or
# modify it.
#
# The PMML package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Please see the
######################################################################################
#
# Author: Max Ferguson
# Date: April 2016
#-------------------------------------------------------------------------------------

import numpy as np
import datetime
import getpass
from lxml import etree as ET
from lxml.etree import Element,SubElement

def trans_get_para(model):
    """Return important parameters used throughout the PMML file"""
    X = model.X
    Y = model.y
    nugget = model.nugget
    k_lambda = model.theta_

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    k_lambda = np.squeeze(k_lambda)
    gamma = 1

    return X,Y,nugget,k_lambda,gamma

def trans_get_dimension(X,Y):
    """Return dimensions of the X and Y arrays"""
    sx = X.shape
    sy = Y.shape

    xrow = sx[0]
    yrow = sy[0]

    if len(sx)==1:
        xcol = 1
    else:
        xcol = sx[1]

    if len(sy)==1:
        ycol = 1
    else:
        ycol = sy[1]

    return xrow,yrow,xcol,ycol



def trans_name(xcol, ycol):
    """Get the names of the columns"""
    featureName = []
    for i in range(xcol):
        featureName.append('x{0}'.format(i+1))

    targetName = []
    for i in range(ycol):
        targetName.append('y{0}'.format(i+1))

    return featureName,targetName



def trans_root(description,copyright,Annotation):
    """Some basic information about the document """
    username = str(getpass.getuser())
    py_version = "0.1"

    PMML_version = "4.3"
    xmlns = "http://www.dmg.org/PMML-4_2"
    PMML = root = Element('pmml',xmlns=xmlns, version=PMML_version)

    # pmml level
    if copyright is None:
        copyright = "Copyright (c) 2015 {0}".format(username)
    if description is None:
        description = "Gaussian Process Model"
    Header = SubElement(PMML,"header",copyright=copyright,description=description)

    if Annotation is not None:
        ET.Element(Header,"Annotation").text=Annotation
    return PMML



def trans_dataDictionary(PMML,featureName,targetName,xcol,ycol):
    """DataField level"""
    toStr = "{0}".format
    DataDictionary = SubElement(PMML,"datadictionary",numberoffields=toStr(xcol+ycol))
    for it_name in featureName:
        SubElement(DataDictionary, "datafield", name=it_name,optype="continuous", datatype="double" )

    for it_name in targetName:
        SubElement(DataDictionary, "datafield", name=it_name,optype="continuous", datatype="double" )

    return PMML



def trans_GP(PMML):
    """Create GaussianProcessModel level"""
    GaussianProcessModel = SubElement(PMML,"gaussianprocessmodel")
    GaussianProcessModel.set("modelname","Gaussian Process Model")
    GaussianProcessModel.set("functionname","regression")
    return GaussianProcessModel



def trans_miningSchema(GaussianProcessModel,featureName,targetName):
    """Create Mining Schema"""
    MiningSchema = SubElement(GaussianProcessModel,"miningschema")
    for it_name in featureName:
        SubElement(MiningSchema, "miningfield", name=it_name,usagetype="active")

    for it_name in targetName:
        SubElement(MiningSchema, "miningfield", name=it_name,usagetype="predicted")

    return GaussianProcessModel



def trans_output(GaussianProcessModel):
    """Create the output level"""
    Output = SubElement(GaussianProcessModel,"output")
    SubElement(Output,"outputfield",name="MeanValue",optype="continuous",datatype="double", feature="predictedValue")
    SubElement(Output,"outputfield",name="StandardDeviation",optype="continuous",datatype="double", feature="predictedValue")
    return GaussianProcessModel



def trans_kernel(GaussianProcessModel,k_lambda,nugget,gamma,xcol,corr):
    """Create Kernel information level"""
    toStr = "{0}".format
    theta = " ".join(map(toStr,k_lambda))

    if "absolute_exponential" in str(corr):
        raise Exception("absolute_exponential not fully supported")
        AbsoluteExponentialKernelType = SubElement(GaussianProcessModel,"AbsoluteExponentialKernelType")
        SubElement(AbsoluteExponentialKernelType,"gamma",value="1")
        SubElement(AbsoluteExponentialKernelType,"noiseVariance",value=toStr(nugget))
        lambda_ = ET.SubElement(AbsoluteExponentialKernelType,"lambda")
        SubElement(lambda_,"Array",n=toStr(xcol),type="real").text=theta

    elif "squared_exponential" in str(corr):
        name = "ardsquaredexponentialkernel"
        attrib = {'gamma':toStr(gamma),'noiseVariance':toStr(nugget)}
        SquaredExponentialKernelType = SubElement(GaussianProcessModel,name,**attrib)
        lambda_ = SubElement(SquaredExponentialKernelType,"lambda")
        SubElement(lambda_,"Array",n=toStr(xcol),type="real").text = theta
    return GaussianProcessModel



def trans_traininginstances(GaussianProcessModel,xrow,xcol):
    """Create traininginstances level"""
    toStr = "{0}".format
    attr = {"recordcount":toStr(xrow),"istransformed":"false","fieldcount":toStr(xcol)}
    GaussianProcessDictionary = SubElement(GaussianProcessModel,"traininginstances",**attr)
    return GaussianProcessDictionary



def trans_instancefields(GaussianProcessDictionary,featureName,targetName):
    """Create instancefields level"""
    InstanceFields = SubElement(GaussianProcessDictionary,"instancefields")
    for name in featureName:
        SubElement(InstanceFields, "instancefield", column=name,field=name)

    for name in targetName:
        SubElement(InstanceFields, "instancefield", column=name,field=name)
    return GaussianProcessDictionary



def trans_inlinetable(GaussianProcessDictionary,featureName,targetName,X,Y):
    """Create the inlinetable for features and targets"""
    toStr = "{0}".format
    InlineTable = SubElement(GaussianProcessDictionary,"inlinetable")
    for i in range(len(Y)):
        Row = SubElement(InlineTable,"row")
        for j,colname in enumerate(featureName):
            Col = SubElement(Row, colname)
            Col.text = toStr(X[i][j])

        for j,colname in enumerate(targetName):
            Col = SubElement(Row, colname)
            Col.text = toStr(Y[i][j])
    return GaussianProcessDictionary
