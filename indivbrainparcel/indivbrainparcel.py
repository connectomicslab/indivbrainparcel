#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from glob import glob
from operator import itemgetter
from datetime import datetime
import argparse
import json
import subprocess
import concurrent.futures
import time


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


class MyBIDs:
    def __init__(self, bids_dir: str):
        fold_list = os.listdir(bids_dir)
        subjids = []
        for it in fold_list:
            if 'sub-' in it:
                subjids.append(it)
        subjids.sort()

        bidsdict = {}

        for subjid in subjids:
            subjdir = os.path.join(bids_dir, subjid)
            sesids = []
            if os.path.isdir(subjdir):
                fold_list = os.listdir(subjdir)
                for it in fold_list:
                    if 'ses-' in it:
                        sesids.append(it)
                sesids.sort()
                bidsdict.__setitem__(subjid, sesids)
        self.value = bidsdict

    def add(self, key, value):
        print(key)
        print(value)
        self[key] = value

    def get_subjids(self, bids_dir: str):

        fold_list = os.listdir(bids_dir)
        subjids = []
        for it in fold_list:
            if 'sub-' in it:
                subjids.append(it)
        subjids.sort()
        self.subjids = subjids
        return subjids

    def get_sesids(self, bids_dir: str, subjid):

        sesids = []
        subjdir = os.path.join(bids_dir, subjid)
        if os.path.isdir(subjdir):
            fold_list = os.listdir(subjdir)
            for it in fold_list:
                if 'ses-' in it:
                    sesids.append(it)
            sesids.sort()
            self.sesids = sesids

        return sesids

# Print iterations progress
def _printprogressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printend="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printend    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlength = int(length * iteration // total)
    bar = fill * filledlength + '-' * (length - filledlength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printend)
    # Print New Line on Complete
    if iteration == total:
        print()


def _build_args_parser():

    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=52)

    from argparse import ArgumentParser

    p = argparse.ArgumentParser(formatter_class=SmartFormatter, description='\n Help \n')

    requiredNamed = p.add_argument_group('Required arguments')
    requiredNamed.add_argument('--subjdir', '-d', action='store', required=True, metavar='SUBJDIR', type=str, nargs=1,
                                help="R| FreeSurfer subjects folder. \n"
                                "\n",
                                default=None)
    
    requiredNamed.add_argument('--outdir', '-o', action='store', required=False, metavar='SUBJDIR', type=str, nargs=1,
                                help="R| Ouput folder. If the folder does not exist then it will be created. \n"
                                " If the ouput folder is not supplied the results will be stored in a ""derivatives"" folder next to the freesurfer subjects directory. \n",
                                default=None)
    
    requiredNamed.add_argument('--subjid', '-s', action='store', required=True,
                                metavar='SUBJID', type=str, nargs=1,
                                help="R| FreeSurfer Subject Identification codes. Please separate the codes by coma.\n"
                                    " - A txt file with the subjects ids can be also supplied. \n",
                                default=None)
    
    requiredNamed.add_argument('--nthreads', '-n', action='store', required=False, metavar='NTHREADS', type=str, nargs=1,
                                help="R| Number of processes to run in parallel. \n", default=['1'])

    requiredNamed.add_argument('--growwm', '-g', action='store', required=False, metavar='GROWWM', type=str, nargs=1,
                                help="R| Grow of GM labels inside the white matter in mm. \n", default=['2'])

    requiredNamed.add_argument('--mixwm', '-m', action='store_true', required=False, default=False,
                                help="R| Mix the cortical WM growing with the cortical GM. This will be used to extend the cortical GM inside the WM. \n"
                                    "\n")
    requiredNamed.add_argument('--remwm', '-rw', action='store_true', required=False, default=False,
                                help="R| Remove the white matter. Set the white matter to 0.  \n"
                                    " This option will not remove the labelled white matter")

    
    p.add_argument('--verbose', '-v', action='store', required=False,
                    type=int, nargs=1,
                    help='verbosity level: 1=low; 2=debug')

    args = p.parse_args()

    if args.subjdir is None :
        print('--subjdir is a REQUIRED arguments')
        sys.exit()

    subjdir = args.subjdir[0]
    
    if not os.path.isdir(subjdir):
        print("\n")
        print("Please, supply a valid FreeSurfer Subjects directory.")
        p.print_help()
        sys.exit()

    return p


def _parc_tsv_table(codes, names, colors, tsv_filename):
    # Table for parcellation
    # 1. Converting colors to hexidecimal string
    seg_hexcol = []
    nrows, ncols = colors.shape
    for i in np.arange(0, nrows):
        seg_hexcol.append(rgb2hex(colors[i, 0], colors[i, 1], colors[i, 2]))

    bids_df = pd.DataFrame(
        {
            'index': np.asarray(codes),
            'name': names,
            'color': seg_hexcol
        }
    )
    #     print(bids_df)
    # Save the tsv table
    with open(tsv_filename, 'w+') as tsv_file:
        tsv_file.write(bids_df.to_csv(sep='\t', index=False))


# Find Structures
def _search_in_atlas(in_atlas, st_tolook, out_atlas, labmax):
    for i, v in enumerate(st_tolook):
        result = np.where(in_atlas == v)
        out_atlas[result[0], result[1], result[2]] = i + labmax + 1
    #         print('%u === %u', v, i + labmax + 1)

    labmax = labmax + len(st_tolook)
    return out_atlas, labmax


# Search the value inside a vector
def search(values, st_tolook):
    ret = []
    for v in st_tolook:
        index = values.index(v)
        ret.append(index)
    return ret


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def hex2rgb(hexcode):
    return tuple(map(ord, hexcode[1:].decode('hex')))

def read_fscolorlut(lutFile):
    # Readind a color LUT file
    fid = open(lutFile)
    LUT = fid.readlines()
    fid.close()

    # Make dictionary of labels
    LUT = [row.split() for row in LUT]
    st_names = []
    st_codes = []
    cont = 0
    for row in LUT:
        if len(row) > 1 and row[0][0] != '#' and row[0][0] != '\\\\':  # Get rid of the comments
            st_codes.append(int(row[0]))
            st_names.append(row[1])
            if cont == 0:
                st_colors = np.array([[int(row[2]), int(row[3]), int(row[4])]])
            else:
                ctemp = np.array([[int(row[2]), int(row[3]), int(row[4])]])
                st_colors = np.append(st_colors, ctemp, axis=0)
            cont = cont + 1

    return st_codes, st_names, st_colors


def _launch_annot2ind(fs_annot, ind_annot, hemi, out_dir, fullid, atlas):

    fssubj_dir = os.environ.get("SUBJECTS_DIR")

    # Creating the hemisphere id
    if hemi == 'lh':
        hemicad = 'L'
    elif hemi == 'rh':
        hemicad = 'R'

    # Moving the Annot to individual space
    subprocess.run(['mri_surf2surf', '--srcsubject', 'fsaverage', '--trgsubject', fullid,
                            '--hemi', hemi, '--sval-annot', fs_annot,
                            '--tval', ind_annot],
                            stdout=subprocess.PIPE, universal_newlines=True)
    


    fs_subj_temp = CorticalParcellation(subjects_dir=fssubj_dir, subjid=fullid, hemi=hemi,
                            annot=ind_annot, surface='white', cortex_label='cortex')

    fs_subj_temp.load_annotation()
    fs_subj_temp.load_surface()
    fs_subj_temp.load_cortex_label()
    fs_subj_temp.correct_parcellation(corr_annot=ind_annot)


    # Copying the resulting annot to the output folder
    out_annot = os.path.join(out_dir, fullid + '_hemi-' + hemicad + '_space-orig_' + atlas + '_dparc.annot')
    subprocess.run(['cp', ind_annot, out_annot], stdout=subprocess.PIPE, universal_newlines=True)

    return out_annot

def _launch_freesurfer(t1file:str, fssubj_dir:str, fullid:str):


    os.environ["SUBJECTS_DIR"] = fssubj_dir

    # Computing FreeSurfer
    subprocess.run(['recon-all', '-subjid', '-i', t1file, fullid, '-all'],
                    stdout=subprocess.PIPE, universal_newlines=True)

    return


def _launch_surf2vol(fssubj_dir, out_dir, fullid, atlas, gm_grow):

    if 'desc' not in atlas:
        atlas_str = atlas + '_desc-'
    else:
        atlas_str = atlas

    if atlas == "aparc":
        atlas_str = "atlas-desikan_desc-aparc"
    elif atlas == "aparc.a2009s":
        atlas_str = "atlas-destrieux_desc-a2009s"

    out_parc = []
    for g in gm_grow:
        out_vol = os.path.join(out_dir, fullid + '_space-orig_' + atlas_str + 'grow' + g + 'mm_dseg.nii.gz')

        if not os.path.isfile(out_vol):
            if g == '0':
                # Creating the volumetric parcellation using the annot files
                subprocess.run(['mri_aparc2aseg', '--s', fullid, '--annot', atlas,
                                '--hypo-as-wm', '--new-ribbon', '--o', out_vol],
                                stdout=subprocess.PIPE, universal_newlines=True)

            else:
                # Creating the volumetric parcellation using the annot files
                subprocess.run(['mri_aparc2aseg', '--s', fullid, '--annot', atlas, '--wmparc-dmax', g, '--labelwm',
                                '--hypo-as-wm', '--new-ribbon', '--o', out_vol],
                                stdout=subprocess.PIPE, universal_newlines=True)


        # Moving the resulting parcellation from conform space to native
        raw_vol = os.path.join(fssubj_dir, fullid, 'mri', 'rawavg.mgz')
        subprocess.run(['mri_vol2vol', '--mov', out_vol, '--targ', raw_vol,
                        '--regheader', '--o', out_vol, '--no-save-reg', '--interp', 'nearest'],
                        stdout=subprocess.PIPE, universal_newlines=True)

        out_parc.append(out_vol)

    return out_parc

def _compute_abased_thal_parc(t1, vol_tparc, deriv_dir, subjid, aseg_nii, out_str):

    cwd = os.path.abspath(os.path.dirname(__file__))
        # Creating data directory
    cwd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cwd, 'data')

    # Cortical atlas directory

    subjid_ent = subjid.split('_')
    subj_id = subjid_ent[0]
    ses_id = subjid_ent[1]
    
    thal_spam = os.path.join(data_dir, 'thalamic_nuclei_MIALatlas', 'Thalamus_Nuclei-HCP-4DSPAMs.nii.gz')
    t1_temp = os.path.join(data_dir, 'mni_icbm152_t1_tal_nlin_asym_09c', 'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz')

    # Creating spatial transformation folder
    stransf_dir = os.path.join(deriv_dir, 'ants-transf2mni', subj_id, ses_id, 'anat')
    if not os.path.isdir(stransf_dir):
        try:
            os.makedirs(stransf_dir)
        except OSError:
            print("Failed to make nested output directory")

    defFile = os.path.join(stransf_dir, subjid + '_space-MNI152NLin2009cAsym_')
    if not os.path.isfile(defFile + 'desc-t12mni_1InverseWarp.nii.gz'):
        # Registration to MNI template
        subprocess.run(['antsRegistrationSyNQuick.sh', '-d', '3', '-f', t1_temp, '-m', t1, '-t', 's',
                        '-o', defFile + 'desc-t12mni_'],
                        stdout=subprocess.PIPE, universal_newlines=True)

    mial_dir = os.path.dirname(vol_tparc)
    # Creating ouput directory
    if not os.path.isdir(mial_dir):
        try:
            os.makedirs(mial_dir)
        except OSError:
            print("Failed to make nested output directory")

    mial_thalparc = os.path.join(mial_dir, subjid + '_space-orig_desc-' + out_str +'_dseg.nii.gz')
    mial_thalspam = os.path.join(mial_dir, subjid + '_space-orig_desc-' + out_str +'_probseg.nii.gz')

    # Applying spatial transform
    subprocess.run(['antsApplyTransforms', '-d', '3', '-e', '3', '-i', thal_spam,
                    '-o', mial_thalspam, '-r', t1, '-t', defFile + 'desc-t12mni_1InverseWarp.nii.gz',
                    '-t','[' + defFile + 'desc-t12mni_0GenericAffine.mat,1]', '-n', 'Linear'],
                    stdout=subprocess.PIPE, universal_newlines=True)

    # Creating MaxProb
    _spams2maxprob(mial_thalspam, 0.05, mial_thalparc, aseg_nii, 10, 49)
    mial_thalparc = [mial_thalparc]

    return mial_thalparc


def _spams2maxprob(spamImage:str, thresh:float=0.05, maxpName:str=None, thalMask:str=None, thl_code:int=10, thr_code:int=49):
    # ---------------- Thalamic nuclei (MIAL) ------------ #
    thalm_codesl       = np.array([1, 2, 3, 4, 5, 6, 7])
    thalm_codesr       = np.array([8, 9, 10, 11, 12, 13, 14])
    thalm_names        =  ['pulvinar', 'ventral-anterior', 'mediodorsal', 'lateral-posterior-ventral-posterior-group', 'pulvinar-medial-centrolateral-group', 'ventrolateral', 'ventral-posterior-ventrolateral-group']
    prefix             = "thal-lh-"
    thalm_namesl       = [prefix + s.lower() for s in thalm_names]
    prefix             = "thal-rh-"
    thalm_namesr       = [prefix + s.lower() for s in thalm_names]
    thalm_colorsl      = np.array([[255,   0,   0], [0, 255,   0], [255, 255, 0], [255, 123, 0], [0, 255, 255], [255, 0, 255], [0, 0, 255]])
    thalm_colorsr      = thalm_colorsl

    # ---------------- Creating output filenames ------------ #
    outDir           = os.path.dirname(spamImage)
    fname            = os.path.basename(spamImage)
    tempList         = fname.split('_')
    tempList[-1]     = 'dseg.nii.gz'
    if not maxpName:
        maxpName         =  os.path.join(outDir, '_'.join(tempList))

    tempList[-1]     = 'dseg.lut'
    lutName          =  os.path.join(outDir, '_'.join(tempList))
    tempList[-1]     = 'dseg.tsv'
    tsvName          =  os.path.join(outDir, '_'.join(tempList))

    maxlist = maxpName.split(os.path.sep)
    tsvlist = tsvName.split(os.path.sep)
    lutlist = lutName.split(os.path.sep)

    # ---------------- Creating Maximum probability Image ------------- #
    # Reading the thalamic parcellation
    spam_Ip          = nib.load(spamImage)
    affine           = spam_Ip.affine
    spam_Ip          = spam_Ip.get_fdata()
    spam_Ip[spam_Ip < thresh] = 0
    spam_Ip[spam_Ip > 1]      = 1

    # 1. Left Hemisphere
    It               = spam_Ip[:, :, :, :7]
    ind              = np.where(np.sum(It, axis=3) == 0)
    maxprob_thl      = spam_Ip[:, :, :, :7].argmax(axis=3) + 1
    maxprob_thl[ind] = 0

    if thalMask:
        Itemp        = nib.load(thalMask)
        Itemp        = Itemp.get_fdata()
        index        = np.where(Itemp != thl_code)
        maxprob_thl[index[0], index[1], index[2]] = 0

    # 2. Right Hemisphere
    It               = spam_Ip[:, :, :, 7:]
    ind              = np.where(np.sum(It, axis=3) == 0)
    maxprob_thr      = spam_Ip[:, :, :, 7:].argmax(axis=3) + 1
    maxprob_thr[ind] = 0

    if thalMask:
        index        = np.where(Itemp != thr_code)
        maxprob_thr[index[0], index[1], index[2]] = 0

    ind              = np.where(maxprob_thr != 0)
    maxprob_thr[ind] = maxprob_thr[ind] + 7

    # Saving the Nifti file
    imgcoll          = nib.Nifti1Image(maxprob_thr.astype('int16') + maxprob_thl.astype('int16'), affine)
    nib.save(imgcoll, maxpName)

    # Creating the corresponding TSV file
    _parc_tsv_table(np.concatenate((thalm_codesl, thalm_codesr)),
                    np.concatenate((thalm_namesl, thalm_namesr)),
                    np.concatenate((thalm_colorsl, thalm_colorsr)),
                    tsvName)

    # Creating and saving the corresponding colorlut table
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    luttable = ['# $Id: <BIDsDirectory>/derivatives/{} {} \n'.format('/'.join(lutlist[-5:]), date_time),
                        '# Corresponding parcellation: ',
                        '# <BIDsDirectory>/derivatives/' + '/'.join(maxlist[-5:]) ,
                        '# <BIDsDirectory>/derivatives/' + '/'.join(tsvlist[-5:]) + '\n']
    luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3} \n '.format("#No.", "Label Name:", "R", "G", "B", "A"))

    luttable.append("# Left Hemisphere. Thalamic nuclei parcellation (MIAL, Najdenovska and Alemán-Gómez et al, 2018)")
    for roi_pos, roi_name in enumerate(thalm_namesl):
        luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(roi_pos + 1, roi_name, thalm_colorsl[roi_pos,0], thalm_colorsl[roi_pos,1], thalm_colorsl[roi_pos,2], 0))
    nright = roi_pos +1

    luttable.append('\n')

    luttable.append("# Right Hemisphere. Thalamic nuclei parcellation")
    for roi_pos, roi_name in enumerate(thalm_namesr):
        luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(nright + roi_pos + 1, roi_name, thalm_colorsr[roi_pos,0], thalm_colorsr[roi_pos,1], thalm_colorsr[roi_pos,2], 0))

    with open(lutName, 'w') as colorLUT_f:
                colorLUT_f.write('\n'.join(luttable))


# Correct the annotation file
class CorticalParcellation:
    """
    Class to load and correct the cortical parcellation from FreeSurfer.

    Parameters
    ----------
    subjid : str
        Subject ID. Default is 'fsaverage'.

    hemi : str
        Hemisphere. Default is 'lh'.

    subjects_dir : str
        Subjects directory. Default is the currect FreeSurfer subjects directory.

    annot : str
        Annotation file. Default is 'aparc'.

    surface : str
        Surface file. Default is 'white'.

    cortex_label : str
        Cortex label file. Default is 'cortex'.

    Returns
    -------
    CorticalParcellation object. It has a method to correct the parcellation.


    """
        

    def __init__(self, subjid: str = 'fsaverage', hemi: str = 'lh', subjects_dir: str = None, annot: str = 'aparc', surface: str = 'white', cortex_label: str = 'cortex'):
        """
        Class to load and visualize cortical parcellations from FreeSurfer.

        Parameters
        ----------
        subjid : str
            Subject ID. Default is 'fsaverage'.

        hemi : str  
            Hemisphere. Default is 'lh'.

        subjects_dir : str
            Subjects directory. Default is the currect FreeSurfer subjects directory.

        annot : str
            Annotation file. Default is 'aparc'.

        surface : str
            Surface file. Default is 'white'.

        cortex_label : str
            Cortex label file. Default is 'cortex'.

        Returns
        -------
        CorticalParcellation object
        """


        # Check if the subject directory exists
        if subjects_dir is None:
            subjects_dir = os.environ['SUBJECTS_DIR']

        # Check if annotation file exists
        if os.path.isfile(annot):
            annot2load = annot
        else:
            # Otherwise, check if the annotation file exists in the subject directory
            annot2load = os.path.join(subjects_dir, subjid, 'label', hemi + '.' + annot + '.annot')
        
        # Check if the surface file exists
        if os.path.isfile(surface):
            surf2load = surface
        else:
            # Otherwise, check if the surface file exists in the subject directory
            surf2load = os.path.join(subjects_dir, subjid, 'surf', hemi + '.' + surface)
        
        # Check if the cortex label file exists
        if os.path.isfile(cortex_label):
            cortex_label2load = cortex_label
        else:
            # Otherwise, check if the cortex label file exists in the subject directory
            cortex_label2load = os.path.join(subjects_dir, subjid, 'label', hemi + '.' + cortex_label + '.label')
        
        self.annot2load = annot2load
        self.surf2load  = surf2load
        self.label2load = cortex_label2load

    def load_annotation(self):
        annot2load = self.annot2load

        # If the annotation file does not exist, raise an error, otherwise load the annotation
        if os.path.isfile(annot2load):
            vert_lab, reg_ctable, reg_names = nib.freesurfer.read_annot(annot2load)
        else:
            raise ValueError('Annotation file not found. Annotation, surface and cortex label files are mandatory to correct the parcellation.')
        
        self.vert_lab = vert_lab
        self.reg_ctable = reg_ctable
        self.reg_names = reg_names


    def load_surface(self):
        surf2load = self.surf2load

        # If the surface file does not exist, raise an error, otherwise load the surface
        if os.path.isfile(surf2load):
            vertices, faces = nib.freesurfer.read_geometry(surf2load)
        else:
            raise ValueError('Surface file not found. Annotation, surface and cortex label files are mandatory to correct the parcellation.')
        
        self.vertices = vertices
        self.faces    = faces
        
    def load_cortex_label(self):

        label2load = self.label2load
        # Check if the cortex label file exists
        
        # If the cortex label file does not exist, raise an error, otherwise load the cortex label
        if os.path.isfile(label2load):
            cortex_label = nib.freesurfer.read_label(label2load)
        else:
            raise ValueError('Cortex label file not found. Annotation, surface and cortex label files are mandatory to correct the parcellation.')
        
        self.cortex_label = cortex_label

    
    def correct_parcellation(self, corr_annot: str = None):
        """
        Correct the parcellation by refilling the vertices from the cortex label file that do not have a label in the annotation file.

        Returns
        -------
        Corrected parcellation

        """
        # Get the vertices from the cortex label file that do not have a label in the annotation file

        faces = self.faces
        vert_lab = self.vert_lab
        vert_lab[vert_lab == -1] = 0

        reg_ctable = self.reg_ctable
        reg_names = self.reg_names
        cortex_label = self.cortex_label


        ctx_lab = vert_lab[cortex_label].astype(int) # Vertices from the cortex label file that have a label in the annotation file

        bool_bound = vert_lab[faces] != 0

        # Boolean variable to check the faces that contain at least two vertices that are different from 0 and at least one vertex that is not 0 (Faces containing the boundary of the parcellation)
        bool_a = np.sum(bool_bound, axis=1) < 3 
        bool_b = np.sum(bool_bound, axis=1) > 0 
        bool_bound = bool_a & bool_b

        faces_bound = faces[bool_bound,:]
        bound_vert = np.ndarray.flatten(faces_bound)

        vert_lab_bound = vert_lab[bound_vert]

        # Delete from the array bound_vert the vertices that contain the vert_lab_bound different from 0
        bound_vert = np.delete(bound_vert, np.where(vert_lab_bound != 0)[0])
        bound_vert = np.unique(bound_vert)

        # Detect which vertices from bound_vert are in the  cortex_label array
        bound_vert = bound_vert[np.isin(bound_vert, cortex_label)]

        bound_vert_orig = np.zeros(len(bound_vert))
        # Create a while loop to fill the vertices that are in the boundary of the parcellation
        # The loop will end when the array bound_vert is empty or when bound_vert is equal bound_vert_orig


        # Detect if the array bound_vert is equal to bound_vert_orig
        bound = np.array_equal(bound_vert, bound_vert_orig)

        while len(bound_vert) > 0:

            if not bound:
                bound_vert_orig = np.copy(bound_vert)
                temp_Tri = np.zeros((len(bound_vert), 100))
                for pos,i in enumerate(bound_vert):
                    # Get the neighbors of the vertex
                    neighbors = np.unique(faces[np.where(faces == i)[0],:])
                    neighbors = np.delete(neighbors, np.where(neighbors == i)[0])
                    temp_Tri[pos,0:len(neighbors)] = neighbors
                temp_Tri = temp_Tri.astype(int)
                index_zero = np.where(temp_Tri == 0)
                labels_Tri = vert_lab[temp_Tri]
                labels_Tri[index_zero] = 0

                for pos,i in enumerate(bound_vert):

                    # Get the labels of the neighbors
                    labels = labels_Tri[pos,:]
                    # Get the most frequent label different from 0
                    most_frequent_label = np.bincount(labels[labels != 0]).argmax()

                    # Assign the most frequent label to the vertex
                    vert_lab[i] = most_frequent_label


                ctx_lab = vert_lab[cortex_label].astype(int) # Vertices from the cortex label file that have a label in the annotation file

                bool_bound = vert_lab[faces] != 0

                # Boolean variable to check the faces that contain at least one vertex that is 0 and at least one vertex that is not 0 (Faces containing the boundary of the parcellation)
                bool_a = np.sum(bool_bound, axis=1) < 3 
                bool_b = np.sum(bool_bound, axis=1) > 0 
                bool_bound = bool_a & bool_b

                faces_bound = faces[bool_bound,:]
                bound_vert = np.ndarray.flatten(faces_bound)

                vert_lab_bound = vert_lab[bound_vert]

                # Delete from the array bound_vert the vertices that contain the vert_lab_bound different from 0
                bound_vert = np.delete(bound_vert, np.where(vert_lab_bound != 0)[0])
                bound_vert = np.unique(bound_vert)

                # Detect which vertices from bound_vert are in the  cortex_label array
                bound_vert = bound_vert[np.isin(bound_vert, cortex_label)]

                bound = np.array_equal(bound_vert, bound_vert_orig)

        # Save the annotation file
        if corr_annot is not None:
            if os.path.isfile(corr_annot):
                os.remove(corr_annot)
            
            # Create folder if it does not exist
            os.makedirs(os.path.dirname(corr_annot), exist_ok=True)
            nib.freesurfer.write_annot(corr_annot, vert_lab, reg_ctable, reg_names)

        return corr_annot, vert_lab, reg_ctable, reg_names
    


# def _build_parcellation(layout, bids_dir, deriv_dir, ent_dict, parccode):
def _build_parcellation(fssubj_dir, subjid, growwm, out_dir, bool_mixwm, bool_rmwm):

    # if fssubj_dir finishes with a / then remove it
    if fssubj_dir[-1] == os.path.sep:
        fssubj_dir = fssubj_dir[:-1]


    cort_dict = {"String":"laus2018",
        "Atlas":"Lausanne",
        "Description":"# 1. Cortical parcellation (L): Lausanne multi-scale cortical parcellation.",
        "Citation":"(Symmetric version of Cammoun et al, 2012)",
        "Type":"annot",
        "Name":["atlas-laus2018_desc-scale1","atlas-laus2018_desc-scale2","atlas-laus2018_desc-scale3","atlas-laus2018_desc-scale4","atlas-laus2018_desc-scale5"],
        "OutSurfLocation":"annots-lausparc",
        "OutVolLocation":"volparc-lausparc"}

    atlas_str     = cort_dict["String"]
    atlas_desc    = cort_dict["Description"]
    atlas_cita    = cort_dict["Citation"]
    atlas_type    = cort_dict["Type"]
    atlas_names   = cort_dict["Name"]

    # Creating data directory
    cwd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cwd, 'data')

    # Cortical atlas directory
    atlas_dir = os.path.join(data_dir, 'ctx_laus2018_sym')

    fs_indivdir = os.path.join(fssubj_dir, subjid)

    if not os.path.isdir(fs_indivdir):
        print(f"The FreeSurfer directory for subject {subjid} does not exist.")
        sys.exit()
    else:
        if out_dir is None:
            temp = os.path.dirname(fssubj_dir)
            
            if os.path.basename(temp) != 'derivatives':
                out_dir = os.path.join(temp, 'derivatives')
            else:
                out_dir = temp
            
        
        subjid_ent = subjid.split('_')
        subj_id = subjid_ent[0]
        ses_id = subjid_ent[1]

        surfatlas_dir = os.path.join(out_dir, cort_dict["OutSurfLocation"], subj_id, ses_id, 'anat')
        volatlas_dir  = os.path.join(out_dir, cort_dict["OutVolLocation"], subj_id, ses_id, 'anat')
        volthal_dir   = os.path.join(out_dir, 'volparc-mialthal', subj_id, ses_id, 'anat')
        
        if not os.path.isdir(surfatlas_dir):
            try:
                os.makedirs(surfatlas_dir)
            except OSError:
                print("Failed to make nested output directory")
        
        if not os.path.isdir(volatlas_dir):
            try:
                os.makedirs(volatlas_dir)
            except OSError:
                print("Failed to make nested output directory")
        
        if not os.path.isdir(volthal_dir):
            try:
                os.makedirs(volthal_dir)
            except OSError:
                print("Failed to make nested output directory")
        
    ######## ------------- Detecting FreeSurfer Subjects Directory  ------------ #
    fshome_dir = os.getenv('FREESURFER_HOME')
    os.environ["SUBJECTS_DIR"] = fssubj_dir

    ######## ------------- Reading FreeSurfer color lut table ------------ #
    lutFile = os.path.join(fshome_dir, 'FreeSurferColorLUT.txt')
    st_codes, st_names, st_colors = read_fscolorlut(lutFile)

    ######## ------------- Labelling the structures ------------ #
    # 1. ---------------- Detecting White matter  ------------------------- #
    wm_codesl = np.array([2, 5001])
    idx = search(st_codes, wm_codesl)
    wm_namesl = ['wm-lh-brain-segmented', 'wm-lh-brain-unsegmented']
    wm_colorsl = st_colors[idx]

    wm_codesr = np.array([41, 5002])
    idx = search(st_codes, wm_codesr)
    wm_namesr = ['wm-rh-brain-segmented', 'wm-rh-brain-unsegmented']
    wm_colorsr = st_colors[idx]

    cc_codes = np.array([250, 251, 252, 253, 254, 255])
    idx = search(st_codes, cc_codes)
    cc_names = np.array(st_names)[idx].tolist()
    prefix = 'wm-brain-'
    cc_names = [prefix + s.lower() for s in cc_names]
    cc_names = [s.replace('_', '-').lower() for s in cc_names]
    cc_colors = st_colors[idx]

    wm_codes = np.concatenate((wm_codesl.astype(int), wm_codesr.astype(int), cc_codes.astype(int)))
    wm_names = ['wm-brain-white_matter']
    wm_colors = np.array([[255, 255, 255]])

    # 2. ---------------- Detection of Thalamic nuclei (MIAL)------------ #
    thalm_codesl = np.array([1, 2, 3, 4, 5, 6, 7])
    thalm_codesr = np.array([8, 9, 10, 11, 12, 13, 14])
    thalm_names = ['pulvinar', 'ventral-anterior', 'mediodorsal', 'lateral-posterior-ventral-posterior-group',
                    'pulvinar-medial-centrolateral-group', 'ventrolateral', 'ventral-posterior-ventrolateral-group']
    prefix = "thal-lh-"
    thalm_namesl = [prefix + s.lower() for s in thalm_names]
    prefix = "thal-rh-"
    thalm_namesr = [prefix + s.lower() for s in thalm_names]
    thalm_colorsl = np.array(
        [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 123, 0], [0, 255, 255], [255, 0, 255], [0, 0, 255]])
    thalm_colorsr = thalm_colorsl

    # 3. ---------------- Detecting Subcortical structures (Freesurfer) ------------------------- #
    subc_codesl = np.array([11, 12, 13, 26, 18, 17, 16])
    idx = search(st_codes, subc_codesl)
    subc_namesl = np.array(st_names)[idx].tolist()
    subc_namesl = [s.replace('Left-', 'subc-lh-').lower() for s in subc_namesl]
    subc_colorsl = st_colors[idx]

    subc_codesr = np.array([50, 51, 52, 58, 54, 53])
    idx = search(st_codes, subc_codesr)
    subc_namesr = np.array(st_names)[idx].tolist()
    subc_namesr = [s.replace('Right-', 'subc-rh-').lower() for s in subc_namesr]
    subc_colorsr = st_colors[idx]
    
    ## ================ Creating the new parcellation
    lut_lines = ['{:<4} {:<40} {:>3} {:>3} {:>3} {:>3} \n \n'.format("#No.", "Label Name:", "R", "G", "B", "A")]
    parc_desc_lines = []

    ##### ========== Selecting the cortical parcellation ============== #####

    parc_desc_lines.append(atlas_desc + ' ' + atlas_cita)

    if not os.path.isfile(os.path.join(fs_indivdir,'mri', 'aparc+aseg.mgz')):
        print(f"FreeSurfer outputs are not complete please run freesurfer for subject {subjid}.")

    else:
        # Finding the cortical parcellations
        out_sparc = glob(os.path.join(surfatlas_dir, subjid + '*' + atlas_str + '*.annot'))

        if len(out_sparc) != len(atlas_names)*2:
            print(f"The selected cortical parcellation ({cort_dict['Atlas']}) is not available for subject {subjid}")
            print("Trying to compute the corresponding cortical parcellation.")

            # Creating the link for fsaverage
            fsave_dir = os.path.join(fshome_dir, 'subjects', 'fsaverage')
            if not os.path.isdir(os.path.join(fssubj_dir, 'fsaverage')):
                process = subprocess.run(['ln', '-s', fsave_dir, fssubj_dir],
                                        stdout=subprocess.PIPE, universal_newlines=True)

            out_sparc = []
            for atlas in atlas_names:

                # Mapping the annot file to individual space
                # 1. Left Hemisphere
                ind_annot     = os.path.join(fssubj_dir, subjid, 'label', 'lh.' + atlas + '.annot') # Annot in individual space (freesurfer subject's directory)
                out_annot = os.path.join(surfatlas_dir, subjid + '_hemi-L_space-orig_' + atlas + '_dparc.annot')
                if not os.path.isfile(out_annot):
                    if atlas_type == 'annot':
                        fs_annot  = os.path.join(atlas_dir,
                                                'lh.' + atlas + '.annot')  # Annot in fsaverage space (Atlas folder)
                        out_annot = _launch_annot2ind(fs_annot, ind_annot, 'lh', surfatlas_dir, subjid, atlas)

                    elif atlas_type == 'gcs':
                        fs_gcs    = os.path.join(atlas_dir,
                                                'lh.' + atlas + '.gcs')  # GCS in fsaverage space (Atlas folder)
                        out_annot = _launch_gcs2ind(fssubj_dir, fs_gcs, ind_annot, 'lh', surfatlas_dir, subjid, atlas)


                out_sparc.append(out_annot) # Annot in individual space (Atlases subject's directory)

                # 2. Right Hemisphere
                ind_annot = os.path.join(fssubj_dir, subjid, 'label', 'rh.' + atlas + '.annot') # Annot in individual space (freesurfer subject's directory)
                out_annot = os.path.join(surfatlas_dir, subjid + '_hemi-R_space-orig_' + atlas + '_dparc.annot')
                if not os.path.isfile(out_annot):
                    if atlas_type == 'annot':
                        fs_annot  = os.path.join(atlas_dir,
                                                'rh.' + atlas + '.annot')  # Annot in fsaverage space (Atlas folder)
                        out_annot = _launch_annot2ind(fs_annot, ind_annot, 'rh', surfatlas_dir, subjid, atlas)

                    elif atlas_type == 'gcs':
                        fs_gcs    = os.path.join(atlas_dir,
                                                'rh.' + atlas + '.gcs')  # GCS in fsaverage space (Atlas folder)
                        out_annot = _launch_gcs2ind(fssubj_dir, fs_gcs, ind_annot, 'rh', surfatlas_dir, subjid, atlas)

                out_sparc.append(out_annot) # Annot in individual space (Atlases subject's directory)


        # Right hemisphere (Surface parcellation)
        rh_cparc = [s for s in out_sparc if "hemi-R" in s]  # Right cortical parcellation
        rh_cparc.sort()

        # Left hemisphere (Surface parcellation)
        lh_cparc = [s for s in out_sparc if "hemi-L" in s]  # Left cortical parcellation
        lh_cparc.sort()

        vol2look = []
        for s in growwm:
            if s.isnumeric():
                vol2look.append('grow' + s + 'mm')
            else:
                vol2look.append('grow' + s)


        vol_cparc = []
        for g in growwm:
            tmp = glob(os.path.join(volatlas_dir, subjid + '*' + atlas_str + '*grow' + g +'*.nii.gz'))  # Cortical surface parcellation (.annot, .gii)
            vol_cparc.extend(tmp)

        # If the volumetric parcellation does not exist it will try to create it
        if len(vol_cparc) != len(atlas_names)*len(growwm):
            vol_cparc = []
            for atlas in atlas_names:
                out_vol = _launch_surf2vol(fssubj_dir, volatlas_dir, subjid, atlas, growwm)
                vol_cparc.extend(out_vol)
        vol_cparc.sort()

        if len(rh_cparc) != len(lh_cparc):  # Verifying the same number of parcellations for both hemispheres
            print(
                "Error: Some surface-based cortical parcellations are missing. Different number of files per hemisphere.\n")
            sys.exit(1)

        for f in rh_cparc:  # Verifying the existence of all surface-based cortical parcellations (Right)
            temp_file = Path(f)
            if not temp_file.is_file():
                print("Error: Some surface-based cortical parcellations are missing.\n")
                sys.exit(1)

        for f in lh_cparc:  # Verifying the existence of all surface-based cortical parcellations (Left)
            temp_file = Path(f)
            if not temp_file.is_file():
                print("Error: Some surface-based cortical parcellations are missing.\n")
                sys.exit(1)

        for f in vol_cparc:  # Verifying the existence of volumetric parcellations
            temp_file = Path(f)
            if not temp_file.is_file():
                print("Error: Volumetric parcellations are missing.\n")
                sys.exit(1)

        # Loading Aparc parcellation
        tempDir = os.path.join(fssubj_dir, subjid)
        aparc_mgz = os.path.join(tempDir, 'mri', 'aseg.mgz')
        raw_mgz = os.path.join(tempDir, 'mri', 'rawavg.mgz')
        aseg_nii = os.path.join(tempDir, 'tmp', 'aseg.nii.gz')
        process = subprocess.run(['mri_vol2vol', '--mov', aparc_mgz, '--targ', raw_mgz, '--regheader', '--o', aseg_nii, '--no-save-reg', '--interp', 'nearest'],
                                stdout=subprocess.PIPE, universal_newlines=True)
        temp_file = Path(aseg_nii)
        if temp_file.is_file():
            aseg_parc = nib.load(temp_file)
            aseg_parc = aseg_parc.get_fdata()
        else:
            print("Error: Cannot create the parcellation because there are missing files.\n")
            sys.exit(1)

        if 'aseg_parc' in locals():
            dim = aseg_parc.shape
        else:
            aseg_parc = nib.load(vol_cparc[0])
            dim = aseg_parc.shape

        outparc_lh = np.zeros((dim[0], dim[1], dim[2]), dtype='int16')  # Temporal parcellation for the left hemisphere
        outparc_rh = np.zeros((dim[0], dim[1], dim[2]), dtype='int16')  # Temporal parcellation for the right hemisphere


        ##### ========== Selecting Thalamic parcellation ============== #####

        # Thalamic parcellation based on Najdenovska et al, 2018
        vol_tparc = os.path.join(volthal_dir, subjid + '_space-orig_desc-MIALThalamicParc_dseg.nii.gz')

        t1mgz = os.path.join(fssubj_dir, subjid, 'mri', 'T1.mgz')
        t1nii = os.path.join(fssubj_dir, subjid, 'tmp', 'T1.nii.gz')
        raw_mgz = os.path.join(fssubj_dir, subjid, 'mri', 'rawavg.mgz')
        subprocess.run(['mri_vol2vol', '--mov', t1mgz, '--targ', raw_mgz, '--regheader', '--o', t1nii, '--no-save-reg', '--interp', 'trilin'],
                        stdout=subprocess.PIPE, universal_newlines=True)
        

        # Computing thalamic nuclei using atlas-based parcellation
        vol_tparc = _compute_abased_thal_parc(t1nii, vol_tparc, out_dir, subjid, aseg_nii, atlas_str)

        # removing the temporal T1.nii.gz
        os.remove(t1nii)

        # Reading the thalamic parcellation
        temp_iparc = nib.load(vol_tparc[0])
        temp_iparc = temp_iparc.get_fdata()

        # Right Hemisphere
        outparc_rh, st_lengtrh = _search_in_atlas(temp_iparc, thalm_codesr, outparc_rh, 0)

        # Left Hemisphere
        outparc_lh, st_lengtlh = _search_in_atlas(temp_iparc, thalm_codesl, outparc_lh, 0)


        ##### ========== Selecting Subcortical parcellation ============== #####

        # Right Hemisphere
        outparc_rh, st_lengtrh = _search_in_atlas(aseg_parc, subc_codesr, outparc_rh, st_lengtrh)

        # Left Hemisphere
        outparc_lh, st_lengtlh = _search_in_atlas(aseg_parc, subc_codesl, outparc_lh, st_lengtlh)



        # Removing temporal aseg image
        os.remove(aseg_nii)

        parc_desc_lines.append("\n")

        # Creating ouput directory

        # Loop around each parcellation
        out_parc = []
        for i in np.arange(0, len(rh_cparc)):
            right_sdata = nib.freesurfer.io.read_annot(rh_cparc[i], orig_ids=False)
            rh_codes = right_sdata[0]
            rh_colors = right_sdata[1][1:, 0:3]
            rh_stnames = right_sdata[2][1:]

            left_sdata = nib.freesurfer.io.read_annot(lh_cparc[i], orig_ids=False)
            lh_codes = left_sdata[0]
            lh_colors = left_sdata[1][1:, 0:3]
            lh_stnames = left_sdata[2][1:]

            fname = os.path.basename(rh_cparc[i])

            temp = fname.split('_')
            scaleid = [s for s in temp if "desc-" in s]  # Detect if the label key exist

            # Selecting the volumetric parcellations for all the wm grow levels
            if scaleid:
                grow_parcs = [s for s in vol_cparc if scaleid[0] in s]
            else:
                grow_parcs = vol_cparc

            nctx_rh = len(rh_stnames)  # Number of cortical regions in the right hemisphere
            nctx_lh = len(lh_stnames)  # Number of cortical regions in the left hemisphere
            nroi_right = nctx_rh + st_lengtrh  # Number of regions in the right hemisphere

            rh_luttable = ["# Right Hemisphere. Cortical Structures"]
            lh_luttable = ["# Left Hemisphere. Cortical Structures"]

            ##### ========== LUT Cortical Surface (Right Hemisphere)============== #####
            # rh_scode, rh_ctab,
            for roi_pos, roi_name in enumerate(rh_stnames):
                temp_name = 'ctx-rh-{}'.format(roi_name.decode("utf-8"))
                rh_luttable.append(
                    '{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(roi_pos + 1, temp_name, rh_colors[roi_pos, 0],
                                                                rh_colors[roi_pos, 1], rh_colors[roi_pos, 2],
                                                                0))
            maxlab_rh = roi_pos + 1

            for roi_pos, roi_name in enumerate(lh_stnames):
                temp_name = 'ctx-lh-{}'.format(roi_name.decode("utf-8"))
                lh_luttable.append(
                    '{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(nroi_right + roi_pos + 1, temp_name,
                                                                lh_colors[roi_pos, 0],
                                                                lh_colors[roi_pos, 1], lh_colors[roi_pos, 2], 0))
            maxlab_lh = nroi_right + roi_pos + 1

            rh_luttable.append('\n')
            lh_luttable.append('\n')


            ##### ========== Selecting Thalamic parcellation ============== #####
            rh_luttable.append("# Right Hemisphere. Thalamic Structures")
            for roi_pos, roi_name in enumerate(thalm_namesr):
                rh_luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(maxlab_rh + roi_pos + 1, roi_name,
                                                                                thalm_colorsr[roi_pos, 0],
                                                                                thalm_colorsr[roi_pos, 1],
                                                                                thalm_colorsr[roi_pos, 2], 0))
            maxlab_rh = maxlab_rh + roi_pos + 1

            lh_luttable.append("# Left Hemisphere. Thalamic Structures")
            for roi_pos, roi_name in enumerate(thalm_namesl):
                lh_luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(maxlab_lh + roi_pos + 1, roi_name,
                                                                                thalm_colorsr[roi_pos, 0],
                                                                                thalm_colorsr[roi_pos, 1],
                                                                                thalm_colorsr[roi_pos, 2], 0))
            maxlab_lh = maxlab_lh + roi_pos + 1

            rh_luttable.append('\n')
            lh_luttable.append('\n')

            ##### ========== Selecting Subcortical parcellation ============== #####
            rh_luttable.append("# Right Hemisphere. Subcortical Structures")
            for roi_pos, roi_name in enumerate(subc_namesr):
                rh_luttable.append('{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(maxlab_rh + roi_pos + 1, roi_name,
                                                                                subc_colorsr[roi_pos, 0],
                                                                                subc_colorsr[roi_pos, 1],
                                                                                subc_colorsr[roi_pos, 2], 0))
            maxlab_rh = maxlab_rh + roi_pos + 1

            lh_luttable.append("# Left Hemisphere. Subcortical Structures")
            for roi_pos, roi_name in enumerate(subc_namesl):
                lh_luttable.append('{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(maxlab_lh + roi_pos + 1, roi_name,
                                                                                subc_colorsl[roi_pos, 0],
                                                                                subc_colorsl[roi_pos, 1],
                                                                                subc_colorsl[roi_pos, 2], 0))
            maxlab_lh = maxlab_lh + roi_pos + 1

            rh_luttable.append('\n')
            lh_luttable.append('\n')

                        # # Volumetric White matter parcellation (vol_wparc)
            wm_luttable = ["# Global White Matter"]
            wm_luttable.append(
                '{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(3000, wm_names[0], wm_colors[0, 0], wm_colors[0, 1],
                                                            wm_colors[0, 2], 0))
            wm_luttable.append('\n')
            wm_luttable.append("# Right Hemisphere. Gyral White Matter Structures")

            # rh_scode, rh_ctab,
            for roi_pos, roi_name in enumerate(rh_stnames):
                temp_name = 'wm-rh-{}'.format(roi_name.decode("utf-8"))
                wm_luttable.append(
                    '{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(roi_pos + 3001, temp_name,
                                                                255 - rh_colors[roi_pos, 0],
                                                                255 - rh_colors[roi_pos, 1],
                                                                255 - rh_colors[roi_pos, 2],
                                                                0))
            wm_luttable.append('\n')

            wm_luttable.append("# Left Hemisphere. Gyral White Matter Structures")
            for roi_pos, roi_name in enumerate(lh_stnames):
                temp_name = 'wm-lh-{}'.format(roi_name.decode("utf-8"))
                wm_luttable.append('{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format(nroi_right + roi_pos + 3001, temp_name,
                                                                                255 - lh_colors[roi_pos, 0],
                                                                                255 - lh_colors[roi_pos, 1],
                                                                                255 - lh_colors[roi_pos, 2], 0))

            for gparc in grow_parcs:
                
                
                fname            = os.path.basename(gparc)
                templist         = fname.split('_')
                tempVar          = [s for s in templist if "desc-" in s]  # Left cortical parcellation
                descid           = tempVar[0].split('-')[1]

                base_id = subjid.split('_')
                base_id.append('space-orig')
                base_id.append('atlas-' + 'ctxLaus2018thalMIAL')
                base_id.append('desc-' + descid)

                # Saving the parcellation
                outparcFilename = os.path.join(volatlas_dir, '_'.join(base_id) + '_dseg.nii.gz')

                if not os.path.isfile(outparcFilename):
                    # Reading the cortical parcellation
                    temp_iparc = nib.load(gparc)
                    affine = temp_iparc.affine
                    temp_iparc = temp_iparc.get_fdata()

                    out_atlas = np.zeros(np.shape(temp_iparc), dtype='uint32')

                    # Adding cortical regions (Right Hemisphere)

                    ind = np.where(np.logical_and(temp_iparc > 2000, temp_iparc < 3000))
                    out_atlas[ind[0], ind[1], ind[2]] = temp_iparc[ind[0], ind[1], ind[2]] - np.ones((len(ind[0]),)) * 2000

                    # Adding the rest of the regions (Right Hemisphere)
                    ind = np.where(outparc_rh > 0)
                    out_atlas[ind[0], ind[1], ind[2]] = outparc_rh[ind[0], ind[1], ind[2]] + np.ones((len(ind[0]),)) * nctx_rh

                    # Adding cortical regions (Left Hemisphere)
                    ind = np.where(np.logical_and(temp_iparc > 1000, temp_iparc < 2000))
                    out_atlas[ind[0], ind[1], ind[2]] = temp_iparc[ind[0], ind[1], ind[2]] - np.ones(
                        (len(ind[0]),)) * 1000 + np.ones(
                        (len(ind[0]),)) * nroi_right

                    # Adding the rest of the regions (Left Hemisphere + Brainstem)
                    ind = np.where(outparc_lh > 0)
                    out_atlas[ind[0], ind[1], ind[2]] = outparc_lh[ind[0], ind[1], ind[2]] + np.ones(
                        (len(ind[0]),)) * nctx_lh + np.ones((len(ind[0]),)) * nroi_right

                    # Adding global white matter
                    bool_ind = np.in1d(temp_iparc, wm_codes)
                    bool_ind = np.reshape(bool_ind, np.shape(temp_iparc))
                    ind      = np.where(bool_ind)
                    out_atlas[ind[0], ind[1], ind[2]] = 3000

                    # Adding right white matter
                    ind      = np.where(np.logical_and(temp_iparc > 4000, temp_iparc < 5000))
                    out_atlas[ind[0], ind[1], ind[2]] = temp_iparc[ind[0], ind[1], ind[2]] - np.ones((len(ind[0]),)) * 1000

                    # Adding left white matter
                    ind = np.where(np.logical_and(temp_iparc > 3000, temp_iparc < 4000))
                    out_atlas[ind[0], ind[1], ind[2]] = temp_iparc[ind[0], ind[1], ind[2]] + np.ones((len(ind[0]),)) * nroi_right

                    if  bool_mixwm:
                        ind      = np.where(out_atlas > 3000)
                        out_atlas[ind[0], ind[1], ind[2]] = out_atlas[ind[0], ind[1], ind[2]] - 3000

                    if  bool_rmwm:
                        ind      = np.where(out_atlas == 3000)
                        out_atlas[ind[0], ind[1], ind[2]] = out_atlas[ind[0], ind[1], ind[2]] - 3000

                    # fname            = os.path.basename(gparc)
                    # templist         = fname.split('_')
                    # tempVar          = [s for s in templist if "desc-" in s]  # Left cortical parcellation
                    # descid           = tempVar[0].split('-')[1]

                    # base_id = subjid.split('_')
                    # base_id.append('space-orig')
                    # base_id.append('atlas-' + 'ctxLaus2018thalMIAL')
                    # base_id.append('desc-' + descid)

                    # # Saving the parcellation
                    # outparcFilename = os.path.join(volatlas_dir, '_'.join(base_id) + '_dseg.nii.gz')
                    imgcoll          = nib.Nifti1Image(out_atlas.astype('int16') , affine)
                    nib.save(imgcoll, outparcFilename)

                    # Saving the colorLUT
                    colorlutFilename = os.path.join(volatlas_dir, '_'.join(base_id) + '_dseg.lut')

                    now              = datetime.now()
                    date_time        = now.strftime("%m/%d/%Y, %H:%M:%S")
                    time_lines       = ['# $Id: {} {} \n'.format(colorlutFilename, date_time),
                                        '# Corresponding parcellation: ',
                                        '# ' + outparcFilename + '\n']

                    hdr_lines        = ['{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}'.format("#No.", "Label Name:", "R", "G", "B", "A")]
                    lut_lines        = time_lines + parc_desc_lines + hdr_lines + rh_luttable + lh_luttable + wm_luttable
                    with open(colorlutFilename, 'w') as colorLUT_f:
                        colorLUT_f.write('\n'.join(lut_lines))

                    st_codes_lut, st_names_lut, st_colors_lut = read_fscolorlut(colorlutFilename)

                    # Saving the TSV
                    tsvFilename = os.path.join(volatlas_dir, '_'.join(base_id) + '_dseg.tsv')
                    _parc_tsv_table(st_codes_lut, st_names_lut, st_colors_lut, tsvFilename)

                    os.remove(gparc)
                
                if os.path.isfile(outparcFilename):
                    out_parc.append(outparcFilename)
                    
    return out_parc    


def main():
    # 0. Handle inputs
    parser = _build_args_parser()
    args = parser.parse_args()

    print(args)
    if args.verbose is not None:
        v = np.int(args.verbose[0])
    else:
        v = 0
        print('- Verbose set to 0\n')
    if v:
        print('\nInputs\n')
    #

    # Getting the path of the current running python file

    fssubj_dir       = args.subjdir[0]
    subjids      = args.subjid[0].split(sep=',')
    growwm       = args.growwm[0]
    nthreads     = int(args.nthreads[0])
    growwm       = growwm.split(',')
    bool_mixwm   = args.mixwm
    bool_rmwm    = args.mixwm
    
    if isinstance(args.outdir, list):
        out_dir      = args.outdir[0]
    else:
        out_dir      = None

    # detect if subjid is a list
    if not isinstance(subjids, list):
        if os.path.isfile(subjids):
            with open(subjids, 'r') as f:
                subjids = f.readlines()
                subjids = [x.strip() for x in subjids]


    nsubj = len(subjids)
        
    if nthreads == 1:
        for i, subjid in enumerate(subjids):
            _printprogressbar(i + 1, nsubj,
                    'Processing T1w --> ' + subjid + ': ' + '(' + str(i + 1) + '/' + str(nsubj) + ')')
                    # _build_parcellation(layout, bids_dir, deriv_dir, ent_dict, parccode)
            _build_parcellation(fssubj_dir, subjid, growwm, out_dir, bool_mixwm, bool_rmwm)
    else:
        start_time = time.perf_counter()
        ncores = os.cpu_count()

        if nthreads > 4:
            nthreads = nthreads - 4
        with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
        #     results = [executor.submit(do_something, sec) for sec in secs]
            results = list(executor.map(_build_parcellation, [fssubj_dir] * nsubj, subjids,
             [growwm] * nsubj, [out_dir] * nsubj), [bool_mixwm] * nsubj, [bool_rmwm] * nsubj)

        end_time = time.perf_counter()

        print(f'Finished in {end_time - start_time} seconds (s)...')


if __name__ == "__main__":
    main()
