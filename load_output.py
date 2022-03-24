"""
@author: wen pan, wenpan@utexas.edu
"""
import numpy as np
class empty():
    pass

def fread(fid, nelements, dtype):

    """Equivalent to Matlab fread function"""

    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array



def read_ecl(filename):
    out = {};

    fid = open(filename,'rb');
    
    # Skip header
    fread(fid,1,'>i4')
#    fread(fid, 1, 'int32=>double', 0, 'b');
    
#    % Read one property at the time
    i = 0;
    while (True):
        i = i + 1;
        
#        % Read field name (keyword) and array size
        keyword=fread(fid, 8, '>i1')
        keyword="".join([chr(item) for item in keyword])
#        keyword = deblank(fread(fid, 8, 'uint8=>char')');
#        keyword = strrep(keyword, '+', '_');
        num=fread(fid, 1, '>i4')[0][0]
#        num = fread(fid, 1, 'int32=>double', 0, 'b');
        
#        % Read and interpret data type
        dtype = fread(fid, 4, '>i1')
        dtype="".join([chr(item) for item in dtype])
        if dtype=='INTE':
            conv =  '>i4';
            wsize = 4;
        elif dtype=='REAL':
                conv = '>f';
                wsize = 4;
        elif dtype=='DOUB':
                conv = '>d';
                wsize = 8;
        elif dtype=='LOGI':
                conv = '>i4';
                wsize = 4;
        elif dtype=='CHAR':
                conv = '>i1';
                num = num * 8;
                wsize = 1;
        else:
            print('data type wrong')

        
#        % Skip next word
        fread(fid, 1, '>i4')
        
#        % Read data array, which may be split into several consecutive
#        % arrays
        data = [];
        remnum = num;
        while remnum > 0:
#            % Read array size
            buflen = fread(fid, 1, '>i4');
            bufnum = int(buflen[0,0] / wsize);
            
#            % Read data and append to array
            data.append(fread(fid, bufnum, conv))#ok<AGROW>
            
#            % Skip next word and reduce counter
            fread(fid, 1, '>i4');
            remnum = remnum - bufnum;
#        end
        
#        % Special post-processing of the LOGI and CHAR datatypes
        data=np.squeeze(np.array(data))
        if dtype=='LOGI':
                data = data>0;
        elif dtype=='CHAR':
                data = np.reshape(data,(8,-1))
                
                
#        % Add array to struct. If keyword already exists, append data.
        if keyword in out.keys():
            out[keyword].append(data);
        else:
            out[keyword] = [data];
#        end
#        % Skip next word
        try:
            fread(fid, 1, '>i4');
        except ValueError:
            print('done')
            break
            

    for key in out.keys():
        out[key]=np.stack(out[key])
    fid.close();
    return out


#%%
#import pandas as pd
#
#dd=read_ecl('D:/commingle/fine_prop/base_3_3_2022/FINE.UNSMRY')
#data=dd['PARAMS  ']