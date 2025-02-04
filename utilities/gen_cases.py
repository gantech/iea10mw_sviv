# coding: utf-8
import numpy as np
import pandas as pd
import yaml, glob, sys, shutil, subprocess
from pathlib import Path
from multiprocessing import Pool

def gen_case(case_data, template_dir='template', case_dir=None):
    """Generate a set of Exawind cases to perform power curve

    Inputs:
       template_dir: Directory containing templates. Expected to be of the form
                     fsi_run
                        - Nalu-wind yaml file
                        - amr-wind input file
                        - static_box.txt
                        - exawind yaml file
                     openfast_run
                        - Main fst input file
                        - ElastoDyn input file
                        - inp.yaml driver input file
       case_data: Dictionary containing case data
                  {'wspd': BLAH, 'yaw': BLAH, 'azimuth': BLAH, 'amplitude': BLAH}
                  Any other dictionary items will be ignored
       case_dir: Name of directory to which the new files will be written

    Outputs:
       None: Files written to directory 'case_dir'
    """

    if ( not Path(template_dir).exists() ):
        print("Template directory ", template, " doesn't exist. Please check your inputs")
        sys.exit()

    wspd = case_data['wspd']
    yaw = case_data['yaw']
    az = case_data['azimuth']
    pitch = case_data['pitch']
    amplitude = case_data['amplitude']

    if (case_dir is None):
        case_dir = 'ws_{:04.1f}_yaw_{:04.1f}_pitch_{:04.1f}_az_{:04.1f}_amp_{:04.1f}/'.format(wspd, yaw, pitch, az, amplitude)


    #First OpenFAST files
    #Path(case_dir+'openfast_run').mkdir(parents=True, exist_ok=True)
    # of_files = glob.glob(template_dir+'/openfast_run/*')
    # for f in of_files:
    #     dest_file = case_dir+'openfast_run/'+f.split('/')[-1]
    shutil.copytree(Path(template_dir+'/openfast_run'), Path(case_dir+'/openfast_run'))
    subprocess.run(["sed", "-i", "s/PITCH_ANGLE/{}/".format(case_data['pitch']), case_dir+'openfast_run/00_IEA-10.0-198-RWT_ElastoDyn.dat' ])
    subprocess.run(["sed", "-i", "s/YAW_ANGLE/{}/".format(case_data['yaw']), case_dir+'openfast_run/00_IEA-10.0-198-RWT_ElastoDyn.dat' ])
    subprocess.run(["sed", "-i", "s/AZIMUTH_ANGLE/{}/".format(case_data['azimuth']), case_dir+'openfast_run/00_IEA-10.0-198-RWT_ElastoDyn.dat' ])

    #Now I need to generate the mode shape for the desired amplitude and put it inside the nalu input file

    #Now Exawind files
    Path(case_dir+'/cfd_run').mkdir(parents=True, exist_ok=True)

    nalu_inp_file = Path(template_dir+'/cfd_run/iea10mw-nalu.yaml')
    with open(nalu_inp_file, 'r') as f:
        nif = yaml.load(f, Loader=yaml.UnsafeLoader)
        nif['realms'][0]['initial_conditions'][0]['value']['velocity'] = [case_data['wspd'], 0.0, 0.0]
        #Entires 0,2,4 are anti-cone, cone and shaft tilt respectively
        nif['realms'][0]['mesh_transformation'][1]['motion'][0]['angle'] = pitch #Pitch
        nif['realms'][0]['mesh_transformation'][3]['motion'][0]['angle'] = az #Azimuth
        nif['realms'][0]['mesh_transformation'][5]['motion'][0]['angle'] = yaw #Yaw
        nif['realms'][0]['mode_shape_analysis']['mode']['amplitude'] = amplitude

        yaml.dump(nif, open(case_dir+'/cfd_run/iea10mw-nalu.yaml','w'), default_flow_style=False)


    shutil.copy( Path(template_dir+'/cfd_run/hypre_file.yaml'), Path(case_dir+'/cfd_run/hypre_file.yaml') )

    amrwind_inp_file = Path(template_dir+'/cfd_run/iea10mw-amr.inp')
    shutil.copy(amrwind_inp_file, Path(case_dir+'/cfd_run/iea10mw-amr.inp'))
    subprocess.run(["sed", "-i", "s/WIND_SPEED/{}/".format(case_data['wspd']), case_dir+'/cfd_run/iea10mw-amr.inp' ])

    exawind_inp_file = Path((template_dir+'/cfd_run/iea10mw.yaml'))
    shutil.copy(exawind_inp_file, Path(case_dir+'/cfd_run/iea10mw.yaml'))
    refinement_box_file = Path(template_dir+'/cfd_run/static_box.txt')
    shutil.copy(refinement_box_file, Path(case_dir+'/cfd_run/static_box.txt'))

if __name__=="__main__":

    case_list = yaml.load(open('case_list.yaml'), Loader=yaml.UnsafeLoader)['siv_viv_cases']
    print(case_list)
    cases = []
    for yc in case_list:
        for amp in yc['amplitude']:
            cases.append( {'wspd': float(yc['ws']),
                           'yaw': float(yc['yaw']),
                           'pitch': float(yc['pitch']),
                           'azimuth': yc['az'],
                           'amplitude': float(amp)} )

    with Pool(36) as p:
        p.map(gen_case, cases)
