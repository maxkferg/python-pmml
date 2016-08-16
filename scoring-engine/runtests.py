import os
import sys
import unittest

root = os.path.dirname(os.path.realpath(__file__))
test = os.path.realpath(os.path.join(root,"../tests"))
sys.path.append(test)
sys.path.append(test)


from test import TestServer

TestServer().test();


#unittest.main()
