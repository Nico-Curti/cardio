# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import shutil
sys.path.append("../modules")
import create_db

def test_create_db():
  sys.path.append("../test_py")
  if os.path.exists("test") == False:
    os.mkdir("test")
  with open("test/test_1_data.txt", 'w') as data:
    data.write("Time,R,G,B,X,Y,Z\n")
    n = 1000
    t = np.arange(n)*0.033
    w = 2*np.pi*np.array([1., 2., 3.])

    # making sinusoidal signals for the red, green and blue signals from
    # t time, w = 2*np.pi*frequency, a amplitude, and h shift from 0
    def mksignal (t, w, a, h):
      s = h
      for _ in w:
        if _ != 0:
          s += a*_**(-2)*np.sin(_*t)
      return s

    r = mksignal(t, w, 0.05, 0.93)
    g = mksignal(t, w, 0.03, 0.13)
    b = mksignal(t, w, 0.02, 0.11)
    x, y, z = np.random.uniform(-0.7, -0.4, n*3).reshape(3, n)
    s = ""
    for _ in np.arange(n):
      s += str(t[_]) + "," + str(r[_]) + "," + str(g[_]) + "," + str(b[_]) +\
        str(x[_]) + "," + str(y[_]) + "," + str(z[_]) + "\n"
    data.write(s)
  with open("test/test_1_info.txt", 'w') as info:
    info.write("Var1,Var2\nDevice,iPhone\nSex,\nAge,\nWeight,\nLength,\nCity,"
               + "\nCountry,\nLifestyle,\nSmoking,\nAfib,\nRhythm,\nClass,\n")
  directory = "test"
  create_db.create_db(directory, directory)
  assert os.path.isfile("cardio.json")
  os.remove("cardio.json")
  shutil.rmtree("test")
