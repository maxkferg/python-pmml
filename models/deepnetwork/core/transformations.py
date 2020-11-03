

class LocalTransformation:
  
  def __init__(self, *args, name=None):
    self.name = name
    self.args = args

  def to_pmml():
    pass

  def apply(self):
    return image


class Preprocessing_imageToTensor(LocalTransformation):
  """Convert an arbitrary tensor to HxWxC format"""

  def apply(self):
    return image



class Preprocessing_crop(LocalTransformation):
  """Center crop an HxWxC image to Hcrop, Wcrop"""

  def apply(self, image):
    return image



class Preprocessing_subtract(LocalTransformation):
  """Subtract a constant from each color channel"""

  def apply(self, image):
    return image



class Preprocessing_divide(LocalTransformation):
  """Divide each color channel by an constant"""

  def apply(self, image):
    return image




def get_transform_function_by_name(name):
  if name == "custom:Preprocessing_imageToTensor":
    return Preprocessing_imageToTensor
  elif name == "crop":
    return Preprocessing_crop
  elif name == "-" or name == "subtract":
    return Preprocessing_subtract
  elif name == "/" or name == "divide":
    return Preprocessing_divide
  else:
    raise ValueError(f"No such function {name}")