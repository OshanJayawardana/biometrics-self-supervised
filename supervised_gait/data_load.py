import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def data_loader_8(path, num_classes, num_sessions, frame_size=50):
    x_train=np.array([])
    y_train=[]
    for user_id in range(1,num_classes+1):
        for session_id in range(1, num_sessions+1):
            try:
                foldername = "u"+str(user_id).rjust(3, '0')+"_w"+str(session_id).rjust(3, '0')
                filename_acc = os.path.join(path,foldername,foldername+"_accelerometer.log")
                filename_gyro = os.path.join(path,foldername,foldername+"_gyroscope.log")
                data_acc = pd.read_csv(filename_acc, header=0, sep="\t")
                #data_gyro = pd.read_csv(filename_gyro, header=0, sep="\t")
                acc_x, acc_y, acc_z = np.array(data_acc.accelerometer_x_data), np.array(data_acc.accelerometer_y_data), np.array(data_acc.accelerometer_z_data)
                #gyro_x, gyro_y, gyro_z = np.array(data_gyro.gyroscope_x_data), np.array(data_gyro.gyroscope_y_data), np.array(data_gyro.gyroscope_z_data)
                #data = [acc_x, acc_y, acc_z, np.sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z),
                #        gyro_x, gyro_y, gyro_z, np.sqrt(gyro_x*gyro_x + gyro_y*gyro_y + gyro_z*gyro_z)]
                #min_ln = min([data[i].shape[0] for i in range(8)])
                #data = [data[i][:min_ln] for i in range(8)]
                
                data = [acc_x, acc_y, acc_z, np.sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z)]
                min_ln = min([data[i].shape[0] for i in range(4)])
                data = [data[i][:min_ln] for i in range(4)]
                
                data = np.array(data).T
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
    unique, counts = np.unique(y_train, return_counts=True)
    weights = dict(zip(unique, (1/counts)/sum(1/counts)))
    return x_train, y_train, weights
    
def data_loader_csv(path, frame_size=130):
  data_acc = pd.read_csv(path)
  acc_x, acc_y, acc_z, acc_mag = np.array(data_acc.accelerometer_x_data), np.array(data_acc.accelerometer_y_data), np.array(data_acc.accelerometer_z_data), np.array(data_acc.a_mag)
  labels = np.array(data_acc.user_id)
  data = [acc_x, acc_y, acc_z, acc_mag, labels]
  data = np.array(data).T
  data=np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
  data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
  #data = data[:(data.shape[0]//frame_size)*frame_size]
  #data=data.reshape(data.shape[0]//frame_size,frame_size,data.shape[1])
  clean_lst=[]
  for i in range(data.shape[0]):
    if len(np.unique(data[i,:,-1]))!=1:
      clean_lst.append(i)
  data = np.delete(data,clean_lst,axis=0)
  x_train = data[:,:,:4]
  y_train = data[:,0,-1]-1
  return x_train, y_train
  
def label_aware_split(x,y,train_split=0.6, test_split=0.5):
  x_train, x_val, x_test, y_train, y_val, y_test = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
  for i in range(np.min(y), np.max(y)+1):
    indx = (y==i).nonzero()[0]
    x_temp = x[indx]
    y_temp = y[indx]
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x_temp, y_temp, test_size=1-train_split)
    x_test_temp, x_val_temp, y_test_temp, y_val_temp = train_test_split(x_test_temp, y_test_temp, test_size=1-test_split)
    if x_train.shape[0]==0:
      x_train, x_val, x_test, y_train, y_val, y_test = x_train_temp, x_val_temp, x_test_temp, y_train_temp, y_val_temp, y_test_temp
    else:
      x_train, y_train = np.concatenate((x_train, x_train_temp), axis=0), np.concatenate((y_train,y_train_temp), axis=0)
      x_val, y_val = np.concatenate((x_val, x_val_temp), axis=0), np.concatenate((y_val, y_val_temp), axis=0)
      x_test, y_test = np.concatenate((x_test, x_test_temp), axis=0), np.concatenate((y_test, y_test_temp), axis=0)
  return x_train, y_train, x_val, y_val, x_test, y_test

def norma(x_all):
  x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
  x=[]
  return x_all