import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

def base_loader(mthd, window_size=128, domain=1):
    x_train=np.array([])
    for atrbt in ['body_acc','total_acc','body_gyro']:
        for axs in ['x','y','z']:
            #path=str(os.path.abspath(os.getcwd()))
            #path+=r"\UCI HAR\uci-human-activity-recognition\original\UCI HAR Dataset\train\Inertial Signals"
            path=r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/"+mthd+"/Inertial Signals"
            file = open(path+r"/"+atrbt+"_"+axs+"_"+mthd+".txt",'r')
            data=(np.array(file.read().split()))
            data=data.reshape(len(data)//window_size,window_size)
            data=data.reshape(data.shape[0],data.shape[1],1)
            data=data.astype(float)
            #data=data[:,:data.shape[1]//2,:]
            if x_train.size==0:
                x_train=data
            else:
                x_train=np.append(x_train,data,axis=2)
    
    if domain:
        file=open(r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/"+mthd+"/Inertial Signals/y_"+mthd+".txt",'r') #for activity recognition
    else:
        file=open(r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/"+mthd+"/Inertial Signals/y_"+mthd+"_sub.txt",'r') #for user authentication
    y_train=np.array(file.read().split())
    y_train=y_train.astype(int)
    y_train=y_train-1
    
    return x_train, y_train

def data_fetch(norma,filt,filt_sigma ,mix=True, domain=1):
    x_train, y_train = base_loader('train', domain=domain)
    x_test, y_test = base_loader('test', domain=domain)
    num_sample=x_train.shape[0]
    x_all = np.concatenate((x_train,x_test), axis=0)
    y_all = np.concatenate((y_train,y_test), axis=0)
    
    if mix:  
      indx = np.arange(x_all.shape[0])
      np.random.shuffle(indx)
      x_all = x_all[indx]
      y_all = y_all[indx]
      
    if filt:
      x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
      x_filt = gaussian_filter1d(x, filt_sigma, axis=1)
      x = x_filt
      x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
    
    if norma:
      x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
      print(x.shape)
      scaler = StandardScaler()
      normalized_x = scaler.fit_transform(x)
      x_all = np.reshape(normalized_x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
      print(x_train.shape)
      
    
    x_train = x_all[:num_sample]
    x_test = x_all[num_sample:]
    
    y_train = y_all[:num_sample]
    y_test = y_all[num_sample:]
    
    return x_train, y_train, x_test, y_test
    
def clas_data_load(samples_per_class, x_train_L, y_train_L, isall=False, domain=1):
    indx = np.arange(x_train_L.shape[0])
    np.random.shuffle(indx)
    x_train_L=x_train_L[indx]
    y_train_L=y_train_L[indx]
    
    if isall:
        x_L = x_train_L[:x_train_L.shape[0]*80//100]
        y_L = y_train_L[:x_train_L.shape[0]*80//100]
        x_L_val = x_train_L[x_train_L.shape[0]*80//100:]
        y_L_val = y_train_L[x_train_L.shape[0]*80//100:]
    
    else:
        if domain:
            cnt=[0,0,0,0,0,0]
            num_labels=6
        else:
            cnt=[0 for i in range(30)]
            num_labels=30
            
        x_L=np.array([])
        y_L=np.array([])
        for i in range(x_train_L.shape[0]):
            for j in range(num_labels):
                if y_train_L[i]==j and cnt[j]<samples_per_class:
                    if x_L.shape[0]==0:
                        x_L=x_train_L[i].reshape((1,128,9))
                    else:
                        x_L=np.append(x_L,x_train_L[i].reshape((1,128,9)),axis=0)
                    y_L=np.append(y_L,j)
                    cnt[j]+=1
            if sum(cnt)>=samples_per_class*num_labels:
                break
        
        print(i)
        
        if domain:
            cnt=[0,0,0,0,0,0]
            num_labels=6
        else:
            cnt=[0 for i in range(30)]
            num_labels=30
            
        x_L_val=np.array([])
        y_L_val=np.array([])
        for k in range(i,x_train_L.shape[0]):
            for j in range(num_labels):
                if y_train_L[k]==j and cnt[j]<(samples_per_class*20/80):
                    if x_L_val.shape[0]==0:
                        x_L_val=x_train_L[k].reshape((1,128,9))
                    else:
                        x_L_val=np.append(x_L_val,x_train_L[k].reshape((1,128,9)),axis=0)
                    y_L_val=np.append(y_L_val,j)
                    cnt[j]+=1
            if sum(cnt)>=(samples_per_class*20/80)*num_labels:
                break
    
    return x_L, y_L, x_L_val, y_L_val              
        
      
    