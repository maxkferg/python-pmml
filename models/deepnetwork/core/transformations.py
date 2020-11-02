


class LocalTransformation:
  
  def __init__(self, etree):
    pass

  def to_pmml():
    pass

  def apply(self):
    pass


class Preprocessing_imageToTensor(LocalTransformation):
  """Convert an arbitrary tensor to HxWxC format"""

  def __init__(self, asrgs):
    pass

  def parse(self, etree):
    pass

  def to_pmml(self)
    pass

  def apply(self):
    pass



class Preprocessing_crop(LocalTransformation):
  """Center crop an HxWxC image to Hcrop, Wcrop"""

  def __init__(self, asrgs):
    pass

  def parse(self, etree):
    pass

  def to_pmml(self)
    pass

  def apply(self):
    pass



class Preprocessing_subtract(LocalTransformation):
  """Subtract a constant from each color channel"""

  def __init__(self, etree):
    pass


  def to_pmml(self):
    pass


  def apply(self):
    pass



class Preprocessing_divide(LocalTransformation):
  """Divide each color channel by an constant"""

  def __init__(self, etree):
    pass


  def to_pmml(self):
    pass


  def apply(self):
    pass




<DerivedField name="imageTensor" optype="continuous" dataType="tensor">
          <Apply function="custom:Preprocessing_imageToTensor">
            <FieldRef field="image"/>
          </Apply>
        </DerivedField>
        <!-- this is to crop the image tensor -->
        <DerivedField name="croppedImageTensor" optype="continuous" dataType="tensor">
            <Apply function="crop">
              <FieldRef field="imageTensor"/>
              <Constant dataType="integer">224</Constant>
              <Constant dataType="integer">224</Constant>
            </Apply>
        </DerivedField>
        <!-- Subtract the color mean -->    
        <DerivedField name="meanNormalizedTensor" optype="continuous" dataType="tensor">
            <Apply function="-">
              <FieldRef field="croppedImageTensor"/>
              <Constant dataType="float">0.485</Constant>
              <Constant dataType="float">0.456</Constant>
              <Constant dataType="float">0.406</Constant>
            </Apply>
        </DerivedField>
        <!-- Divid by color standard deviation -->  
        <DerivedField name="normalizedTensor" optype="continuous" dataType="tensor">
            <Apply function="/">
              <FieldRef field="meanNormalizedTensor"/>
              <Constant dataType="float">0.229</Constant>
              <Constant dataType="float">0.224</Constant>
              <Constant dataType="float">0.225</Constant>
            </Apply>
        </DerivedField>