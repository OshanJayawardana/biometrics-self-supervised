import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_loader(path, num_classes, frame_size=50):
    x_train=np.array([])
    y_train=[]
    for user_id in range(1,num_classes+1):
        for session_id in range(1,16):
            try:
                filename = "u"+str(user_id).rjust(3, '0')+"_w"+str(session_id).rjust(3, '0')+"_data_user_coord.csv"
                data = pd.read_csv(os.path.join(path,filename))
                data = np.array(data)
                data=np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
                data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
                if x_train.shape[0]==0:
                    x_train  = data
                    y_train += [user_id-1]*data.shape[0]
                else:
                    x_train  = np.concatenate((x_train,data), axis=0)
                    y_train += [user_id-1]*data.shape[0]
            except FileNotFoundError:
                continue
    indx = np.arange(len(y_train))
    y_train = np.array(y_train)
    np.random.shuffle(indx)
    x_train = x_train[indx]
    y_train = y_train[indx]
    return x_train, y_train

def norma(x_all):
  x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
  x=[]
  return x_all