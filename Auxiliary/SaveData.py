

import os
import pickle

def SaveData( Data, Names, Dirs, Name ):
    
    for ii in range(0, len(Data) ):
        dat         = Data[ii]
        name        = Names[ii]
        direction   = Dirs[ii]


        if not os.path.exists(direction):
            os.makedirs(direction)
        
        Directions = next(os.walk('./' + direction))[1]
        Nfolder = -1
        for folder in Directions:
            if folder[0:len(Name)] == Name:
                Nfolder     = max( int( folder.split(Name)[1]), Nfolder )
        dirName = direction +'/' + Name + str( Nfolder + 1 )
        os.makedirs(dirName)
        
        IterVarsFile =   dirName + '/' + name + '.pkl'
        with open(IterVarsFile, 'wb') as f:
            pickle.dump(dat, f)
        print("results saved at:" + IterVarsFile)
            

        
    
    