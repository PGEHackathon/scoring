import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# from load_output import fread,read_ecl

sub_id='0' # team name
model_shape=[256,200]
my_test_folder='D:\\commingle\\fine_prop\\submission\\'
my_test_folder2='D:/commingle/fine_prop/submission/'


# read input

def generate_input_deck(file:str, repo_name:str):

    myinput=pd.read_csv(file)
    x=(myinput['X(ft)'].values-20-33333)//40
    y=(myinput['Y(ft)'].values-20-22222)//40

    # correct if well location exceed model size
    x[x>=model_shape[0]]=model_shape[0]-1
    x[x<=0]=0
    y[y>=model_shape[1]]=model_shape[1]-1
    y[y<=0]=0
    completion=myinput['Unit'].values

    # change input
    f=open('template.DATA')
    flist=f.readlines()
    f.close()

    for i in range(3):
        flist[80+i]='WP5'+str(i)+' G '+str(x[i]+1)+' '+str(model_shape[1]-y[i])+' 1400 OIL /\n'
        if completion[i]=='Upper':
            flist[135+i]='WP5'+str(i)+' '+str(x[i]+1)+' '+str(model_shape[1]-y[i])+' 1 25 OPEN 2* 0.316 1* 200 /\n' # upper zone        
        else:
            flist[135+i]='WP5'+str(i)+' '+str(x[i]+1)+' '+str(model_shape[1]-y[i])+' 25 50 OPEN 2* 0.316 1* 200 /\n' # upper zone        

    f=open(f'input_decks/{repo_name}.DATA','w+')
    f.writelines(flist)
    f.close()

# run simulation
# command="\"eclrun eclipse "+my_test_folder+"sub_"+str(sub_id)+".DATA\""
# command='cmd /c '+command
# os.system(command)

# read output
# field_prod=[]
# well_prod=[]
# out=read_ecl(my_test_folder2+'sub_'+str(sub_id)+'.UNSMRY')
# field_prod.append(out['PARAMS  '][-1,2]) # field production




