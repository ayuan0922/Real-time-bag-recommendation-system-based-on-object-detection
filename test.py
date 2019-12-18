import os

fpath=os.path.dirname(__file__)
print(fpath)
fpath1=fpath+'/image'
if not os.path.isdir(fpath1):
    os.mkdir(fpath1)
a=os.path.join(fpath,'image/0.jpg')
print(a)