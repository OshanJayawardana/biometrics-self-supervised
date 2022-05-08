import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
  
def data_load_origin(path, users, folders, frame_size=30):
  sessions = []
  x_train=np.array([])
  y_train=[]
  for user_id,user in enumerate(users):
    count=0
    for folder in folders:
      for session in range(1,6):
        for typ in (["fav", "same"]):
          filename = "user"+str(user)+"_"+typ+"_session"+str(session)+".csv"
          filepath = os.path.join(path, folder, filename)
          try:
            file = pd.read_csv(filepath)
            data = np.array(file.iloc[:, 1:25])
            data = np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
            data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
            if data.shape[2]!=24:
              continue
            if x_train.shape[0]==0:
              x_train  = data
              y_train += [user_id]*data.shape[0]
            else:
              x_train  = np.concatenate((x_train,data), axis=0)
              y_train += [user_id]*data.shape[0]
            #print(filepath)
            count+=1
          except (FileNotFoundError, IndexError):
            continue
    sessions.append(count)
  #print(x_train.shape)
  return x_train, np.array(y_train), sessions
  
def norma_origin(x_all):
  x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
  x=[]
  return x_all
  
def user_data_split(x,y, samples_per_user):
  users, counts  = np.unique(y, return_counts=True)
  x_train = np.array([])
  y_train = np.array([])
  for user in users:
    indx = np.where(y == user)[0]
    np.random.shuffle(indx)
    indx = indx[:samples_per_user]
    if x_train.shape[0]==0:
      x_train = x[indx]
      y_train = y[indx]
    else:
      x_train = np.concatenate((x_train,x[indx]), axis=0)
      y_train = np.concatenate((y_train,y[indx]), axis=0)
  return x_train, y_train
    

def data_load(path, users, frame_size=30):
  sessions = []
  x_train=np.array([])
  x_val=np.array([])
  x_test=np.array([])
  y_train=[]
  y_val=[]
  y_test=[]
  for user_id,user in enumerate(users):
    count=0
    for folder in (["TestingSet", "TestingSet_secret", "TrainingSet"]):
      for session in range(1,6):
        for typ in (["fav", "same"]):
          filename = "user"+str(user)+"_"+typ+"_session"+str(session)+".csv"
          filepath = os.path.join(path, folder, filename)
          try:
            file = pd.read_csv(filepath)
            data = np.array(file.iloc[:, 1:25])
            data = data[:(data.shape[0]//frame_size)*frame_size]
            comp_num_samp = data.shape[0]//frame_size
            samp_train = (comp_num_samp//10)*7
            samp_val = (comp_num_samp//100)*85
            train_data = data[: samp_train*frame_size]
            val_data = data[samp_train*frame_size: samp_val*frame_size]
            test_data = data[samp_val*frame_size:]
            train_data = np.lib.stride_tricks.sliding_window_view(train_data, (frame_size, train_data.shape[1]))[::frame_size//2, :]
            val_data = np.lib.stride_tricks.sliding_window_view(val_data, (frame_size, val_data.shape[1]))[::frame_size//2, :]
            test_data = np.lib.stride_tricks.sliding_window_view(test_data, (frame_size, test_data.shape[1]))[::frame_size//2, :]
            data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
            if data.shape[2]!=24:
              continue
            if x_train.shape[0]==0:
              x_train  = train_data
              y_train += [user_id]*train_data.shape[0]
              
              x_val  = val_data
              y_val += [user_id]*val_data.shape[0]
              
              x_test  = test_data
              y_test += [user_id]*test_data.shape[0]
            else:
              x_train  = np.concatenate((x_train,train_data), axis=0)
              y_train += [user_id]*train_data.shape[0]
              
              x_val  = np.concatenate((x_val,val_data), axis=0)
              y_val += [user_id]*val_data.shape[0]
              
              x_test  = np.concatenate((x_test,test_data), axis=0)
              y_test += [user_id]*test_data.shape[0]
            count+=1
          except (FileNotFoundError, IndexError):
            continue
    sessions.append(count)
  print(x_train.shape)
  return x_train, np.array(y_train), x_val, np.array(y_val), x_test, np.array(y_test), sessions
  
def norma(x_train, x_val, x_test):
  x = np.reshape(x_train,(x_train.shape[0]*x_train.shape[1],x_train.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_train = np.reshape(x,(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
  
  x = np.reshape(x_val,(x_val.shape[0]*x_val.shape[1],x_val.shape[2]))
  x = scaler.transform(x)
  x_val = np.reshape(x,(x_val.shape[0],x_val.shape[1],x_val.shape[2]))
  
  x = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2]))
  x = scaler.transform(x)
  x_test = np.reshape(x,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
  
  return x_train, x_val, x_test

def data_loader_gait(path, classes, frame_size=128):
    x_train=np.array([])
    y_train=[]
    x_val=np.array([])
    y_val=[]
    x_test=np.array([])
    y_test=[]
    sessions=[]
    for user_id, user in enumerate(classes):
        sess_count=0
        for session_id in range(1,2):
            try:
                if user==40:
                  foldername = "u"+str(user).rjust(3, '0')+"_w"+str(2).rjust(3, '0')
                else:
                  foldername = "u"+str(user).rjust(3, '0')+"_w"+str(session_id).rjust(3, '0')
                filename_acc = os.path.join(path,foldername,foldername+"_accelerometer.log")
                data_acc = pd.read_csv(filename_acc, header=0, sep="\t")
                acc_x, acc_y, acc_z = np.array(data_acc.accelerometer_x_data), np.array(data_acc.accelerometer_y_data), np.array(data_acc.accelerometer_z_data)
                data = [acc_x, acc_y, acc_z, np.sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z)]
                min_ln = min([data[i].shape[0] for i in range(4)])
                data = [data[i][:min_ln] for i in range(4)]
                data = np.array(data).T

                data_train = data[:int(data.shape[0]*0.7)]
                data_val = data[int(data.shape[0]*0.7):int(data.shape[0]*0.85)]
                data_test = data[int(data.shape[0]*0.85):]
                data = 0

                data_train=np.lib.stride_tricks.sliding_window_view(data_train, (frame_size,data_train.shape[1]))[::frame_size//2, :]
                data_train=data_train.reshape(data_train.shape[0],data_train.shape[2],data_train.shape[3])

                data_val=np.lib.stride_tricks.sliding_window_view(data_val, (frame_size,data_val.shape[1]))[::frame_size, :]
                data_val=data_val.reshape(data_val.shape[0],data_val.shape[2],data_val.shape[3])

                data_test=np.lib.stride_tricks.sliding_window_view(data_test, (frame_size,data_test.shape[1]))[::frame_size, :]
                data_test=data_test.reshape(data_test.shape[0],data_test.shape[2],data_test.shape[3])

                if x_train.shape[0]==0:
                    x_train  = data_train
                    y_train += [user_id]*data_train.shape[0]
                    x_val  = data_val
                    y_val += [user_id]*data_val.shape[0]
                    x_test  = data_test
                    y_test += [user_id]*data_test.shape[0]
                else:
                    x_train  = np.concatenate((x_train,data_train), axis=0)
                    y_train += [user_id]*data_train.shape[0]

                    x_val = np.concatenate((x_val,data_val), axis=0)
                    y_val += [user_id]*data_val.shape[0]

                    x_test = np.concatenate((x_test,data_test), axis=0)
                    y_test += [user_id]*data_test.shape[0]
                sess_count+=1
            except FileNotFoundError:
                continue
        sessions.append(sess_count)
    indx = np.arange(len(y_train))
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    np.random.shuffle(indx)
    x_train = x_train[indx]
    y_train = y_train[indx]
    return x_train, y_train, x_val, y_val, x_test, y_test, sessions
    
def data_loader_gait_pre(path, classes, frame_size=128):
    x_train=np.array([])
    y_train=[]
    sessions=[]
    for user_id, user in enumerate(classes):
        sess_count=0
        for session_id in range(1,2):
            try:
                foldername = "u"+str(user_id).rjust(3, '0')+"_w"+str(session_id).rjust(3, '0')
                filename_acc = os.path.join(path,foldername,foldername+"_accelerometer.log")
                data_acc = pd.read_csv(filename_acc, header=0, sep="\t")
                acc_x, acc_y, acc_z = np.array(data_acc.accelerometer_x_data), np.array(data_acc.accelerometer_y_data), np.array(data_acc.accelerometer_z_data)
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
                sess_count+=1
            except FileNotFoundError:
                continue
        sessions.append(sess_count)
    indx = np.arange(len(y_train))
    y_train = np.array(y_train)
    np.random.shuffle(indx)
    x_train = x_train[indx]
    y_train = y_train[indx]
    return x_train, y_train, sessions