import os 
import glob
import re
import numpy as np
from matplotlib import ticker


result = glob.glob('*.{}'.format('csv'))

#print(result)


from os import listdir

dir = os.getcwd()
def find_csv(path_to_dir=None,suffix='.csv'):    #By default, find all csv in current working directory 
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and filename.startswith('var1') and 'Experimental' not in filename and 'One Electron Reduction' not in filename and 'Analysis' not in filename and "var" in filename]

def find_experimental_csv(path_to_dir=None,preffix = 'Experimental',suffix='.csv'):  # By default, find all csv starts with experimenatl 
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and 'Experimental' in filename]



def find_sigma(CV):
    CV = CV.replace('.csv','')
    pattern = re.compile(r'var1=([\d.]+[eE][+-][\d]+)')

    var1 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var2=([\d.]+[eE][+-][\d]+)')

    var2 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var3=([\d.]+[eE][+-][\d]+)')

    var3 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var4=([\d.]+[eE][+-][\d]+)')
    
    var4 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var5=([\d.]+[eE][+-][\d]+)')
    
    var5 = float(pattern.findall(CV)[0])
    
    pattern = re.compile(r'var6=([\d.]+[eE][+-][\d]+)')
    
    var6 = float(pattern.findall(CV)[0])
    return var1,var2,var3,var4,var5,var6



def find_conc(CV):
    CV = CV.replace('.csv','')
    pattern = re.compile(r'Point=([A-Z])')

    point = pattern.findall(CV)[0]

    pattern = re.compile(r'Theta=([-]?[\d.]+[eE][+-][\d]+)')
    
    Theta = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var1=([\d.]+[eE][+-][\d]+)')

    var1 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var2=([\d.]+[eE][+-][\d]+)')

    var2 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var3=([\d.]+[eE][+-][\d]+)')

    var3 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var4=([\d.]+[eE][+-][\d]+)')
    
    var4 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var5=([\d.]+[eE][+-][\d]+)')
    
    var5 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var6=([\d.]+[eE][+-][\d]+)')
    
    var6 = float(pattern.findall(CV)[0])

    return point, Theta, var1,var2,var3,var4,var5,var6



def finddir(basedir='.'):
    dirs = [ os.path.join(basedir,name) for name in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, name)) ]
    return dirs

"""
def find_point(CVs:'The collection of CVs',point):
    pattern=re.compile(f'Point={point}.*')

    match = []

    for CV in CVs:
        m = pattern.findall(CV)
        if m is not None and len(m) > 0:
            match.append(m[0])

    return match
"""





if __name__ == "__main__":
    dirs = finddir('../Series Learning')
    print(dirs)



def format_func_dimensionla_potential(value,tick_number):
    #convert dimensionless potential to dimensional potential in mV
    value = value / 96485 * 8.314*298.0 *1e3
    return (f'{value:.2f}')


@ticker.FuncFormatter
def format_func_dimensionless_kf(value,tick_number):
    value = np.log10(10**value * 1e-6*1e-6/1e-9)
    return (f'{value:.1f}')


@ticker.FuncFormatter
def format_func_dimensionless_keq(value,tick_number):
    value = np.log10(10**value/1e-3)
    return (f'{value:.1f}')  

@ticker.FuncFormatter
def format_func_pct(value,tick_number):
    return (f'{value:.1%}')  


@ticker.FuncFormatter
def format_func_nA(value,tick_number):
    return (f'{value*1e9:.0f}')  

