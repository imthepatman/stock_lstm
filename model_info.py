from model import *
import sys

if(len(sys.argv)==2) :
    model_name = sys.argv[1]

    model = Model(model_name)
    model.load()

    print(model.model.summary())
else:
    print("Wrong number of arguments. Exiting.")
