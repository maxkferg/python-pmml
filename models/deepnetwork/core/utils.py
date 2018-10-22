import requests
from lxml import etree


def to_bool(xml):
    if type(xml) is bool:
        return xml
    if xml.lower()=="true":
        return True
    return False


def url_exists(path):
    """
    Return True if a url exists
    """
    r = requests.head(path)
    return r.status_code == requests.codes.ok


def strip_namespace(tree):
    """
    Strip the namespace from an XML tree
    """
    for _, el in tree:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
        for at in el.attrib.keys(): # strip namespaces of attributes too
            if '}' in at:
                newat = at.split('}', 1)[1]
                el.attrib[newat] = el.attrib[at]
                del el.attrib[at]
    return tree


def read_array(Element):
    """
    Read values from an XML array
    """
    array = Element.find("Array")
    items = array.text.split(" ")
    dtype = array.attrib["type"]
    if dtype=="int":
        return [int(i) for i in items]
    elif dtype=="real":
        return [float(i) for i in items]
    elif dtype=="string":
        return [str(i) for i in items]
    else:
        raise ValueError("Unknown datatype %s"%dtype)


def Array(children, dtype="int"):
    """
    Factory for Array element.
    Does not extend Element directly due to cython compatibilty issue
    Returns an Etree Element that meets the PMML array spec
    @tag: The XML tag
    @children: A list of children to add to the array
    @dtype: The array element type. Can be "real" or "int"
    """
    n = str(len(children))
    children = [str(child) for child in children]
    dtypes = ["int","real","string"]
    if dtype not in dtypes:
        raise ValueError("Unknown Array dtype %s"%dtype)
    # Construct the element
    element = etree.Element("Array", n=n, type=dtype)
    element.text = " ".join(children)
    return element
