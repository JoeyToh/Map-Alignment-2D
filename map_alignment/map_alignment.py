'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''
from __future__ import print_function

# import sys
# if sys.version_info[0] == 3:
#     from importlib import reload
# elif sys.version_info[0] == 2:
#     pass

# new_paths = [
#     u'../arrangement/', # this would be relative address from the runMe.y script that loads this module
# ]
# for path in new_paths:
#     if not( path in sys.path):
#         sys.path.append( path )

import time
import copy
import itertools
# import collections

import cv2
import numpy as np
import numpy.linalg
import scipy
import scipy.ndimage
import sympy as sym
import networkx as nx
import sklearn.cluster
import skimage.transform

import matplotlib.transforms
import matplotlib.path as mpath

import Polygon#, Polygon.IO

import arrangement.arrangement as arr

# from map_alignment import utilities as utils
# from map_alignment import mapali_plotting as maplt # this is used in the runMe.py

from . import utilities as utils
from . import mapali_plotting as maplt # this is used in the runMe.py


import multiprocessing as mp
import contextlib as ctx
from functools import partial

################################################################################
######################################################################
################################################################################

def _lock_n_load(file_name, config={}):
    '''
    # loading image:
    loads the image pointed to by the file_name

    NOTE on binarization of image, it happens twice:
    1) input for SKIZ and distance tranforms; where unexplored areas are treated
    as occupied area, because we are only interested SKIZ and distance transform
    of open space.
    2) trait detection; unexplored areas are treated as occupied area, because
    we are only interested in occupied cells for line detection.

    results: 'image', 'traits', 'skiz', 'distance'
    '''
    ### default values
    if 'trait_detection_source' not in config: config['trait_detection_source'] = 'binary_inverted'
    if 'binary_thresholding_1' not in config: config['binary_thresholding_1'] = 200
    if 'binary_thresholding_2' not in config: config['binary_thresholding_2'] = [100, 255]
    if 'edge_detection_config' not in config: config['edge_detection_config'] = [50, 150, 3]
    if 'peak_detect_sinogram_config' not in config: config['peak_detect_sinogram_config'] = [10, 15, 0.15]
    if 'orthogonal_orientations' not in config: config['orthogonal_orientations'] = True
    results = {}

    ######################################## laoding image
    img = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )
    results['image'] = img

    tic = time.time()

    ######################################## get skiz and distance image
    ## NOTE: this is the first binary conversion (see documentation of the method)
    thr = config['binary_thresholding_1']
    # bin_ = np.array( np.where( img < thr, 0, 255 ), dtype=np.uint8)
    _, bin_ = cv2.threshold(img, thr, 255 , cv2.THRESH_BINARY)
    img_skiz, img_disance = _skiz_bitmap(bin_, invert=True, return_distance=True)

    # scaling distance image to [0, 255]
    img_disance += img_disance.min()
    img_disance *= 255. / img_disance.max()

    results['skiz'] = img_skiz
    results['distance'] = img_disance

    ######################################## detect geometric traits
    if 'trait_file_name' in config:
        trait_file_name = config['trait_file_name']

        file_ext = trait_file_name.split('.')[-1]
        if file_ext in ['yaml', 'YAML', 'yml', 'YML']:
            trait_data = arr.utls.load_data_from_yaml( trait_file_name )
            traits = trait_data['traits']
        else:
            raise(StandardError('loading traits from file only supports yaml (svg to come soon)'))

    else:
        ########## detect dominant orientation
        orientations = _find_dominant_orientations(img, config['orthogonal_orientations'])

        ########## get the appropriate input image
        if 'trait_detection_source' not in config:
            print ('WARNING: \'trait_detection_source\' is not defined, original is selected')
            image = img

        elif config['trait_detection_source'] == 'original':
            image = img

        elif config['trait_detection_source'] == 'binary':
            ## NOTE: this is the second binary conversion (see documentation of the method)
            [thr1, thr2] = config['binary_thresholding_2']
            ret, bin_img = cv2.threshold(img, thr1,thr2 , cv2.THRESH_BINARY)
            image = bin_img

        elif config['trait_detection_source'] == 'binary_inverted':
            ## NOTE: this is the second binary conversion (see documentation of the method)
            [thr1, thr2] = config['binary_thresholding_2']
            ret, bin_img = cv2.threshold(img , thr1,thr2 , cv2.THRESH_BINARY_INV)
            image = bin_img

        elif config['trait_detection_source'] == 'edge':
            [thr1, thr2] = config['binary_thresholding_2']
            print ('apt_size is set hard coded')
            apt_size = 3
            edge_img = cv2.Canny(img, thr1, thr2, apt_size)
            image = edge_img

        traits = _find_lines_with_radiography(image, orientations, config['peak_detect_sinogram_config'])


    ########## triming lines to segments, based on boundary
    boundary = [-20, -20, image.shape[1]+20, image.shape[0]+20]
    traits = arr.utls.unbound_traits(traits)
    traits = arr.utls.bound_traits(traits, boundary)
    results['traits'] = traits

    elapsed_time = time.time() - tic

    return results, elapsed_time

################################################################################
###################################################### "signal processing" stuff
################################################################################

################################################################################
def _skiz_bitmap (image, invert=True, return_distance=False):
    '''
    Skeleton of Influence Zone [AKA; Generalized Voronoi Diagram (GVD)]

    Input
    -----
    Bitmap image (occupancy map)
    occupied regions: low value (black), open regions: high value (white)

    Parameter
    ---------
    invert: Boolean (default:False)
    If False, the ridges will be high (white) and backgroud will be low (black)
    If True, the ridges will be low (black) and backgroud will be high (white)

    Output
    ------
    Bitmap image (Skeleton of Influence Zone)


    to play with
    ------------
    > the threshold (.8) multiplied with grd_abs.max()
    grd_binary_inv = np.where( grd_abs < 0.8*grd_abs.max(), 1, 0 )
    > Morphology of the input image
    > Morphology of the grd_binary_inv
    > Morphology of the skiz
    '''

    # original = image.copy()

    ###### image erosion to thicken the outline
    kernel = np.ones((9,9),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    # image = cv2.medianBlur(image, 5)
    # image = cv2.GaussianBlur(image, (5,5), 0).astype(np.uint8)
    # image = cv2.erode(image, kernel, iterations = 1)
    image = cv2.medianBlur(image, 5)

    ###### compute distance image
    # dis = scipy.ndimage.morphology.distance_transform_bf( image )
    # dis = scipy.ndimage.morphology.distance_transform_cdt( image )
    # dis = scipy.ndimage.morphology.distance_transform_edt( image )
    dis = cv2.distanceTransform(image, cv2.DIST_L2,  maskSize=cv2.DIST_MASK_PRECISE)

    ###### compute gradient of the distance image
    dx = cv2.Sobel(dis, cv2.CV_64F, 1,0, ksize=5)
    dy = cv2.Sobel(dis, cv2.CV_64F, 0,1, ksize=5)
    grd = dx - 1j*dy
    grd_abs = np.abs(grd)

    # at some points on the skiz tree, the abs(grd) is very weak
    # this erosion fortifies those points
    kernel = np.ones((3,3),np.uint8)
    grd_abs = cv2.erode(grd_abs, kernel, iterations = 1)

    # only places where gradient is low
    grd_binary_inv = np.where( grd_abs < 0.8*grd_abs.max(), 1, 0 )

    ###### skiz image
    # sometimes the grd_binary_inv becomes high value near the boundaries
    # the erosion of the image means to extend the low-level boundaries
    # and mask those undesired points
    kernel = np.ones((5,5),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    skiz = (grd_binary_inv *image ).astype(np.uint8)

    # ###### map to [0,255]
    # skiz = (255 * skiz.astype(np.float)/skiz.max()).astype(np.uint8)

    ###### sometimes border lines are marked, I don't link it!
    # skiz[:,0] = 0
    # skiz[:,skiz.shape[1]-1] = 0
    # skiz[0,:] = 0
    # skiz[skiz.shape[0]-1,:] = 0


    # ###### post-processing
    kernel = np.ones((3,3),np.uint8)
    skiz = cv2.dilate(skiz, kernel, iterations = 1)
    skiz = cv2.erode(skiz, kernel, iterations = 1)
    skiz = cv2.medianBlur(skiz, 3)
    # kernel = np.ones((3,3),np.uint8)
    # skiz = cv2.dilate(skiz, kernel, iterations = 1)

    ###### inverting
    if invert:
        thr1,thr2 = [127, 255]
        ret, skiz = cv2.threshold(skiz , thr1,thr2 , cv2.THRESH_BINARY_INV)


    if return_distance:
        return skiz, dis
    elif not(return_distance):
        return skiz

################################################################################
def _find_dominant_orientations(image,
                                orthogonal_orientations=False,
                                find_peak_conf={} ):
    '''
    intense smoothing of the image and weithed histogram of gradient is very
    cruicial, min value for filter size for image processing should be 9
    and for 1D signal smoothing atleast 11

    '''
    # flipud: why? I'm sure it won't work otherwise, but dont know why
    # It should happen in both "_find_dominant_orientations" & "find_grid_lines"
    img = np.flipud(image)

    ### smoothing
    img = cv2.blur(img, (11,11))
    img = cv2.GaussianBlur(img, (11,11),0)

    ### oriented gradient of the image
    # is it (dx - 1j*dy) or (dx + 1j*dy)
    # this is related to "flipud" problem mentioned above
    dx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=11)
    dy = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=11)
    grd = dx - 1j*dy
    # return grd


    ### weighted histogram of oriented gradients (over the whole image)
    hist, binc = utils.wHOG (grd, NumBin=180*5, Extension=False)
    hist = utils.smooth(hist, window_len=21)
    # return hist, binc

    ### finding peaks in the histogram
    if orthogonal_orientations:
        # if dominant orientations are orthogonal, find only one and add pi/2 to it
        peak_idx = np.argmax(hist)
        orientations = [binc[peak_idx], binc[peak_idx]+np.pi/2]

    else:
        # setting cwt_peak_detection parameters
        CWT = True if ('CWT' not in find_peak_conf) else find_peak_conf['CWT']
        cwt_range = (5,50,5) if ('cwt_range' not in find_peak_conf) else find_peak_conf['cwt_range']
        Refine_win = 20 if ('Refine_win' not in find_peak_conf) else find_peak_conf['Refine_win']
        MinPeakDist = 30 if ('MinPeakDist' not in find_peak_conf) else find_peak_conf['MinPeakDist']
        MinPeakVal = .2 if ('MinPeakVal' not in find_peak_conf) else find_peak_conf['MinPeakVal']
        Polar = False if ('Polar' not in find_peak_conf) else find_peak_conf['Polar']

        peak_idx = utils.FindPeaks( hist,
                                    CWT=True, cwt_range=(5,50,5),
                                    Refine_win=20 , MinPeakDist = 30 , MinPeakVal=.2,
                                    Polar=False )

        orientations = list(binc[peak_idx])

    # shrinking the range to [-pi/2, pi/2]
    for idx in range(len(orientations)):
        if orientations[idx]< -np.pi/2:
            orientations[idx] += np.pi
        elif np.pi/2 <= orientations[idx]:
            orientations[idx] -= np.pi

    # removing similar angles
    for idx in range(len(orientations)-1,-1,-1):
        for jdx in range(idx):
            if np.abs(orientations[idx] - orientations[jdx]) < np.spacing(10**10):
                orientations.pop(idx)
                break

    ### returning the results
    return np.array(orientations)

################################################################################
def _find_lines_with_radiography(image, orientations, peak_detect_config):
    '''
    '''
    if len(orientations)==0: raise(StandardError('No dominant orientation is available'))
    if peak_detect_config is None: peak_detect_config = [10, 15, 0.15]

    ### fetching setting for sinogram peak detection
    [refWin, minDist, minVal] = peak_detect_config

    ### radiography
    # flipud: why? I'm sure it won't work otherwise, but dont know why
    # It should happen in both "_find_dominant_orientations" & "find_grid_lines"
    image = np.flipud(image)
    imgcenter = (image.shape[1]/2, # cols == x
                 image.shape[0]/2) # rows == y

    sinog_angles = orientations - np.pi/2 # in radian
    sinograms = skimage.transform.radon(image, theta=sinog_angles*180/np.pi )#, circle=True)
    sinogram_center = len(sinograms.T[0])/2

    lines = []
    for (orientation, sinog_angle, sinogram) in zip(orientations, sinog_angles, sinograms.T):
        # Find peaks in sinogram:
        peakind = utils.FindPeaks(utils.smooth(sinogram, window_len=11),
                                  CWT=False, cwt_range=(1,8,1),
                                  Refine_win = int(refWin),
                                  MinPeakDist = int(minDist),
                                  MinPeakVal = minVal,
                                  Polar=False)

        # line's distance to the center of the image
        dist = np.array(peakind) - sinogram_center

        pts_0 = [ ( imgcenter[0] + d*np.cos(sinog_angle),
                    imgcenter[1] + d*np.sin(sinog_angle) )
                  for d in dist]

        pts_1 = [ ( point[0] + np.cos(orientation) ,
                    point[1] + np.sin(orientation) )
                  for point in pts_0]

        lines += [arr.trts.LineModified( args=( sym.Point(p0[0],p0[1]), sym.Point(p1[0],p1[1]) ) )
                  for (p0,p1) in zip(pts_0,pts_1)]

    return lines



################################################################################
######################################################################
################################################################################
def _construct_arrangement(data, config={}):
    '''
    '''
    ########## setting default values

    if 'multi_processing' not in config: config['multi_processing'] = 4
    if 'end_point' not in config: config['end_point'] = False
    if 'timing' not in config: config['timing'] = False

    if 'prune_dis_neighborhood' not in config: config['prune_dis_neighborhood'] = 2
    if 'prune_dis_threshold' not in config: config['prune_dis_threshold'] = .15 # home:0.15 - office:0.075

    # any cell with value below this is considered occupied
    if 'occupancy_threshold' not in config: config['occupancy_threshold'] = 200


    tic = time.time()
    ######################################## deploying arrangement
    # if print_messages: print ('\t deploying arrangement ... ')
    # tic_ = time.time()
    arrange = arr.Arrangement(data['traits'], config)
    # print ('arrangment time:{:.5f}'.format(time.time()-tic_))

    ###############  distance based edge pruning
    # if print_messages: print ('\t arrangement pruning ... ')
    # tic_= time.time()
    _set_edge_distance_value(arrange, data['distance'], config['prune_dis_neighborhood'])
    # print ('_set_edge_distance_value:{:.5f}'.format(time.time()-tic_))
    # tic_= time.time()
    arrange = _prune_arrangement_with_distance(arrange, data['distance'],
                                                     neighborhood=config['prune_dis_neighborhood'],
                                                     distance_threshold=config['prune_dis_threshold'])
    # print ('arrangment pruning:{:.5f}'.format(time.time()-tic_))

    ############### counting occupied cells in each face
    # if print_messages: print ('\t counting occupancy of faces ... ')
    # tic_= time.time()
    arrange = _set_face_occupancy_attribute(arrange, data['image'], config['occupancy_threshold'])
    # print ('counting occupancy of faces:{:.5f}'.format(time.time()-tic_))

    ########################################
    # if print_messages: print ('\t setting ombb attribute of faces ...')
    # tic_ = time.time()
    arrange = _set_ombb_of_faces (arrange)
    # print ('setting ombb attribute of faces:{:.5f}'.format(time.time()-tic_))
    ########################################
    # if print_messages: print ('\t caching face area weight ...')
    # tic_ = time.time()
    superface = arrange._get_independent_superfaces()[0]
    arrange_area = superface.get_area()
    for face in arrange.decomposition.faces:
        face.attributes['area_weight'] = float(face.get_area()) / float(arrange_area)
    # print ('face area weight:{:.5f}'.format(time.time()-tic_))

    elapsed_time = time.time() - tic
    return arrange, elapsed_time


################################################################################
def _set_edge_distance_value(arrangement, distance_image, neighborhood):
    ''''''
    for (s,e,k) in arrangement.graph.edges(keys=True):
        index_set = (s,e,k)
        neighbors = _pixel_neighborhood_of_halfedge (arrangement, index_set,
                                                    neighborhood=neighborhood,
                                                    image_size=distance_image.shape)

        neighbors_val = distance_image[neighbors[:,1], neighbors[:,0]]
        neighbors_val = neighbors_val[~np.isnan(neighbors_val)]
        arrangement.graph[s][e][k]['obj'].attributes['distances'] = neighbors_val


################################################################################
def _prune_arrangement_with_distance(arrangement, distance_image,
                                     neighborhood=2,
                                     distance_threshold=.075):
    '''
    '''

    _set_edge_distance_value(arrangement, distance_image, neighborhood=neighborhood)

    # get edges_to_prun
    forbidden_edges  = arrangement.get_boundary_halfedges()

    edges_to_purge = []
    for (s,e,k) in arrangement.graph.edges(keys=True):

        # rule 1: (self and twin) not in forbidden_edges
        not_forbidden = (s,e,k) not in forbidden_edges
        # for a pair of twin half-edge, it's possible for one to be in forbidden list and the other not
        # so if the occupancy suggests that they should be removed, one of them will be removed
        # this is problematic for arrangement._decompose, I can't let this happen! No sir!
        (ts,te,tk) = arrangement.graph[s][e][k]['obj'].twinIdx
        not_forbidden = not_forbidden and  ((ts,te,tk) not in forbidden_edges)

        # rule 2: low_edge_occupancy - below "low_occ_percent"
        neighbors_dis_val = arrangement.graph[s][e][k]['obj'].attributes['distances']
        sum_ = neighbors_dis_val.sum() / 255.
        siz_ = np.max([1,neighbors_dis_val.shape[0]])
        ok_to_prun_sum = float(sum_)/siz_ > distance_threshold
        # var_ = np.var(neighbors_dis_val)
        # ok_to_prun_var = var_ < 200

        if not_forbidden and ok_to_prun_sum:
            edges_to_purge.append( (s,e,k) )

    # prunning
    arrangement.remove_edges(edges_to_purge, loose_degree=2)
    arrangement.remove_nodes(nodes_idx=[], loose_degree=2)

    return arrangement


################################################################################
def _generate_hypothese(src_arr, src_img_shape,
                       dst_arr, dst_img_shape,
                       config={}):
    '''
    finds transformations between mbb of faces that are not outliers
    similar_labels (default: ignore)
    if similar_labels to be used, "connectivity_maps" should be passed here

    parameters for "_align_ombb()":
    - tform_type='affine'

    parameters for "_reject_implausible_transformations()":
    - scale_mismatch_ratio_threshold (default: 0.5)
    - scale_bounds (default: [.3, 3])

    '''

    if 'scale_mismatch_ratio_threshold' not in config: config['scale_mismatch_ratio_threshold'] = .5
    if 'scale_bounds' not in config: config['scale_bounds'] = [.3, 3] #[.1, 10]

    if 'face_occupancy_threshold' not in config: config['face_occupancy_threshold'] = .5
    # if '' not in config: config[''] =

    tic = time.time()

    tforms = []
    for face_src in src_arr.decomposition.faces:
        # check if src_face is not occupied
        src_occ, src_tot = face_src.attributes['occupancy']
        if (src_occ/src_tot) < config['face_occupancy_threshold']:
            for face_dst in dst_arr.decomposition.faces:
                # check if dst_face is not occupied
                dst_occ, dst_tot = face_dst.attributes['occupancy']
                if (dst_occ/dst_tot) < config['face_occupancy_threshold']:
                    tforms.extend (_align_ombb(face_src,face_dst, tform_type='affine'))

    tforms = np.array(tforms)
    # if print_messages: print ( '\t totaly {:d} transformations estimated'.format(tforms.shape[0]) )
    tforms_total = tforms.shape[0]

    tforms = _reject_implausible_transformations( tforms,
                                                  src_img_shape, dst_img_shape,
                                                  config['scale_mismatch_ratio_threshold'],
                                                  config['scale_bounds'] )
    elapsed_time = time.time() - tic
    tforms_after_reject = tforms.shape[0]

    # if print_messages: print ( '\t and {:d} transformations survived the rejections...'.format(tforms.shape[0]) )
    # if tforms.shape[0] == 0: raise (StandardError('no transformation survived.... '))

    return tforms, elapsed_time, tforms_total, tforms_after_reject # , (total_t, after_reject)


################################################################################
def _select_winning_hypothesis(src_arr, dst_arr, tforms, config={}):
    '''
    input:
    arrangements, tforms

    too_many_tforms = 1000
    sklearn.cluster.DBSCAN(eps=0.051, min_samples=2)

    output:
    hypothesis
    '''

    if 'multiprocessing' not in config: config['multiprocessing'] = True
    if 'too_many_tforms' not in config: config['too_many_tforms'] = 1000
    if 'dbscan_eps' not in config: config['dbscan_eps'] = 0.051
    if 'dbscan_min_samples' not in config: config['dbscan_min_samples'] = 2
    # if '' not in config: config[''] =

    tic = time.time()
    if tforms.shape[0] < config['too_many_tforms']:
        # if print_messages: print ('only {:d} tforms are estimated, so no clustering'.format(tforms.shape[0]))
        if config['multiprocessing']:
            _arrangement_match_score_par = partial( _arrangement_match_score_4partial,
                                                    arrangement_src=src_arr,
                                                    arrangement_dst=dst_arr,
                                                    tforms = tforms)

            with ctx.closing(mp.Pool(processes=4)) as p:
                arr_match_score = p.map( _arrangement_match_score_par, range(len(tforms)))

            best_idx = np.argmax(arr_match_score)

        else:
            arr_match_score = {}
            for idx, tf in enumerate(tforms):
                arrange_src = src_arr
                arrange_dst = dst_arr
                arr_match_score[idx] = _arrangement_match_score(arrange_src, arrange_dst, tf)
                # if print_messages: print ('match_score {:d}/{:d}: {:.4f}'.format(idx+1,tforms.shape[0], arr_match_score[idx]))
                best_idx = max(arr_match_score, key=arr_match_score.get)

        hypothesis = tforms[best_idx]
        n_cluster = 0

    else:
        ################################ clustering transformations and winner selection
        #################### feature scaling (standardization)
        parameters = np.stack([ np.array( [tf.translation[0], tf.translation[1], tf.rotation, tf.scale[0] ] )
                                for tf in tforms ], axis=0)
        assert not np.any( np.isnan(parameters))

        # note that the scaling is only applied to parametes (for clustering),
        # the transformations themselves are intact
        parameters -= np.mean( parameters, axis=0 )
        parameters /= np.std( parameters, axis=0 )

        #################### clustering pool into hypotheses
        # if print_messages: print ('\t clustering {:d} transformations'.format(parameters.shape[0]))
        cls = sklearn.cluster.DBSCAN(eps=config['dbscan_eps'], min_samples=config['dbscan_min_samples'])
        cls.fit(parameters)
        labels = cls.labels_
        unique_labels = np.unique(labels)
        # if print_messages: print ( '\t *** total: {:d} clusters...'.format(unique_labels.shape[0]-1) )

        ###  match_score for each cluster

        if config['multiprocessing']:
            cluster_representative = {}
            for lbl in np.setdiff1d(unique_labels,[-1]):
                class_member_idx = np.nonzero(labels == lbl)[0]
                class_member = [ tforms[idx] for idx in class_member_idx ]

                # pick the one that is closest to all others in the same group
                params = parameters[class_member_idx]
                dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
                dist_arr = dist_mat.sum(axis=0)
                tf = class_member[ np.argmin(dist_arr) ]
                cluster_representative[lbl] = tf

            _arrangement_match_score_par = partial( _arrangement_match_score_4partial,
                                                    arrangement_src=src_arr,
                                                    arrangement_dst=dst_arr,
                                                    tforms = cluster_representative)

            # note that keys to a dictionary (not guaranteed to be ordered ) are passes as idx
            # therefor, later, the index to max arg, is the index to keys of the dictionary
            with ctx.closing(mp.Pool(processes=4)) as p:
                arr_match_score = p.map( _arrangement_match_score_par, cluster_representative.keys())

            max_idx = np.argmax(arr_match_score)
            winning_cluster_key = list(cluster_representative.keys())[max_idx]
            winning_cluster = [tforms[idx] for idx in np.nonzero(labels==winning_cluster_key)[0]]

        else:
            arr_match_score = {}
            for lbl in np.setdiff1d(unique_labels,[-1]):
                class_member_idx = np.nonzero(labels == lbl)[0]
                class_member = [ tforms[idx] for idx in class_member_idx ]

                # pick the one that is closest to all others in the same group
                params = parameters[class_member_idx]
                dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
                dist_arr = dist_mat.sum(axis=0)
                tf = class_member[ np.argmin(dist_arr) ]

                arrange_src = src_arr
                arrange_dst = dst_arr
                arr_match_score[lbl] = _arrangement_match_score(arrange_src, arrange_dst, tf)
                # if print_messages: print ('match score cluster {:d}/{:d}: {:.4f}'.format(lbl,len(unique_labels)-1, arr_match_score[lbl]) )

                ### pick the winning cluster
                winning_cluster_idx = max(arr_match_score, key=arr_match_score.get)
                winning_cluster = [tforms[idx] for idx in np.nonzero(labels==winning_cluster_idx)[0]]

        ### match_score for the entities of the winning cluster
        ### there will be too few to cluster here
        arr_match_score = {}
        for idx, tf in enumerate(winning_cluster):
            arrange_src = src_arr
            arrange_dst = dst_arr
            arr_match_score[idx] = _arrangement_match_score(arrange_src, arrange_dst, tf)
            # if print_messages: print ('match score element {:d}/{:d}: {:.4f}'.format(idx,len(winning_cluster)-1, arr_match_score[idx]) )

        ### pick the wining cluster
        hypothesis_idx = max(arr_match_score, key=arr_match_score.get)
        hypothesis =  winning_cluster[hypothesis_idx]

        n_cluster = unique_labels.shape[0]-1

    # np.save('arr_match_score_'+'_'.join(keys)+'.npy', arr_match_score)
    elapsed_time = time.time() - tic

    return hypothesis, n_cluster, elapsed_time


################################################################################
def _arrangement_match_score_4partial(idx, arrangement_src, arrangement_dst, tforms):
    return _arrangement_match_score(arrangement_src, arrangement_dst, tforms[idx])

################################################################################
def _reject_implausible_transformations(transformations,
                                        image_src_shape, image_dst_shape,
                                        scale_mismatch_ratio_threshold=.1,
                                        scale_bounds=[.1, 10] ):
    '''
    Input
    -----
    transformations (numpy array)
    each element is a skimage.transform.AffineTransform object
    (affine transforms, ie. tf.scale is a tuple)

    image_src_shape, image_dst_shape
    (assuming images are vertically aligned with x-y axes)

    Parameters
    ----------
    scale_mismatch_ratio_threshold (default: .1)
    scale_bounds (default: [.1, 10])
    '''

    ### rejecting transformations that contain nan
    NaN_free_idx = []
    for idx,tf in enumerate(transformations):
        if not np.any( np.isnan( tf.params )):
            NaN_free_idx.append(idx)
    # print ( 'nan_reject: {:d}'.format( len(transformations)-len(NaN_free_idx) ) )
    transformations = transformations[NaN_free_idx]

    ### reject transformations with mismatching scale or extrem scales
    correct_scale_idx = []
    for idx,tf in enumerate(transformations):
        if np.abs(tf.scale[0]-tf.scale[1])/np.mean(tf.scale) < scale_mismatch_ratio_threshold:
            if (scale_bounds[0] < tf.scale[0] < scale_bounds[1]):
                correct_scale_idx.append(idx)
    # print ( 'scale_reject: {:d}'.format( len(transformations)-len(correct_scale_idx) ) )
    transformations = transformations[correct_scale_idx]

    ### reject transformations, if images won't overlap under the transformation
    src_h, src_w = image_src_shape
    src = np.array([ [0,0], [src_w,0], [src_w,src_h], [0,src_h], [0,0] ])

    dst_h, dst_w = image_dst_shape
    dst = np.array([ [0,0], [dst_w,0], [dst_w,dst_h], [0,dst_h], [0,0] ])
    dst_path = _create_mpath(dst)

    overlapping_idx = []
    for idx,tf in enumerate(transformations):
        src_warp = tf._apply_mat(src, tf.params)
        src_warp_path = _create_mpath(src_warp)
        if src_warp_path.intersects_path(dst_path,filled=True):
            overlapping_idx.append(idx)
    # print ('non_overlapping_reject: {:d}'.format(len(transformations)-len(overlapping_idx) ) )

    transformations = transformations[overlapping_idx]

    return transformations

################################################################################
def _arrangement_match_score(arrangement_src, arrangement_dst, tform):
    '''
    tform:  a transformation instance of skimage lib
    '''

    arrange_src = arrangement_src
    arrange_dst = arrangement_dst

    # construct a matplotlib transformation instance (for transformation of paths )
    aff2d = matplotlib.transforms.Affine2D( tform.params )

    ### transforming paths of faces, and updating centre points
    faces_src = arrange_src.decomposition.faces
    faces_dst = arrange_dst.decomposition.faces


    ### find face to face association
    faces_src_path = [face.path.transformed(aff2d) for face in faces_src]
    faces_dst_path = [face.path for face in faces_dst]

    faces_src_poly = [Polygon.Polygon(path.to_polygons()[0]) for path in faces_src_path]
    faces_dst_poly = [Polygon.Polygon(path.to_polygons()[0]) for path in faces_dst_path]

    face_area_src = np.array([poly.area() for poly in faces_src_poly])
    face_area_dst = np.array([poly.area() for poly in faces_dst_poly])
    # cdist expects 2d arrays as input, so I just convert the 1d area value
    # to 2d vectors, all with one (or any aribitrary number)
    face_area_src_2d = np.stack((face_area_src, np.ones((face_area_src.shape))),axis=1)
    face_area_dst_2d = np.stack((face_area_dst, np.ones((face_area_dst.shape))),axis=1)
    f2f_distance = scipy.spatial.distance.cdist(face_area_src_2d,
                                                face_area_dst_2d,
                                                'euclidean')

    face_cen_src = np.array([poly.center() for poly in faces_src_poly])
    face_cen_dst = np.array([poly.center() for poly in faces_dst_poly])

    f2f_association = {}
    for src_idx in range(f2f_distance.shape[0]):
        # if the centre of faces in dst are not inside the current face of src
        # their distance are set to max, so that they become illegible
        # TODO: should I also check for f_dst.path.contains_point(face_cen_src)

        contained_in_src = faces_src_path[src_idx].contains_points(face_cen_dst)
        contained_in_dst = [path_dst.contains_point(face_cen_src[src_idx])
                            for path_dst in faces_dst_path]
        contained = np.logical_and( contained_in_src, contained_in_dst)
        if any(contained):
            maxDist = f2f_distance[src_idx,:].max()
            distances = np.where(contained,
                                 f2f_distance[src_idx,:],
                                 np.repeat(maxDist, contained.shape[0] ))
            dst_idx = np.argmin(distances)
            f2f_association[src_idx] = dst_idx
        else:
            # no face of destination has center inside the current face of src
            pass

    ### find face to face match score (of associated faces)
    f2f_match_score = {(f1_idx,f2f_association[f1_idx]): None
                       for f1_idx in f2f_association.keys()}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        poly_src = faces_src_poly[f1_idx]
        poly_dst = faces_dst_poly[f2_idx]

        union = poly_src | poly_dst
        intersection = poly_src & poly_dst

        # if one of the faces has the area equal to zero
        if union.area() == 0:
            score =  0.
        else:
            overlap_ratio = intersection.area() / union.area()
            overlap_score = (np.exp(overlap_ratio) - 1) / (np.e-1)
            score = overlap_score

        f2f_match_score[(f1_idx,f2_idx)] = score


    ### find the weights of pairs of associated faces to arrangement match score
    face_pair_weight = {}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        # f1_area = faces_src[f1_idx].get_area()
        # f2_area = faces_dst[f2_idx].get_area()

        # f1_w = float(f1_area) / float(arrange_src_area)
        # f2_w = float(f2_area) / float(arrange_dst_area)

        # face_pair_weight[(f1_idx,f2_idx)] = np.min([f1_w, f2_w])
        face_pair_weight[(f1_idx,f2_idx)] = np.min([faces_src[f1_idx].attributes['area_weight'],
                                                    faces_dst[f2_idx].attributes['area_weight']])

    ### computing arrangement match score
    arr_score = np.sum([face_pair_weight[(f1_idx,f2_idx)]*f2f_match_score[(f1_idx,f2_idx)]
                        for (f1_idx,f2_idx) in f2f_match_score.keys()])

    # ################################# Experimental: to be removed
    # # this should normalize match score for when the size of arrangements are not the same
    # src_arr_area = arrangement_src._get_independent_superfaces()[0].get_area() * np.mean(tform.scale)
    # dst_arr_area = arrangement_dst._get_independent_superfaces()[0].get_area()
    # ratio = max([src_arr_area,dst_arr_area]) / min([src_arr_area,dst_arr_area])
    # arr_score *= ratio
    # ################################# Experimental: to be removed
    return arr_score


################################################################################
########################################## "oriented minimum bounding box" stuff
################################################################################
def _set_ombb_of_faces (arrangement):
    '''
    move to arr.utls
    '''
    for face in arrangement.decomposition.faces:
        ombb = _oriented_minimum_bounding_box(face.path.vertices)
        face.attributes['ombb_path'] = _create_mpath(ombb)
    return arrangement

################################################################################
def _align_ombb(face_src,face_dst, tform_type='similarity'):
    '''
    move to arr.utls
    '''
    src = face_src.attributes['ombb_path'].vertices[:-1,:]
    dst = face_dst.attributes['ombb_path'].vertices[:-1,:]

    alignments = [ skimage.transform.estimate_transform( tform_type, np.roll(src,-roll,axis=0), dst )
                   for roll in range(4) ]
    return alignments


################################################################################
def _distance2point(p1,p2,p):
    '''
    move to arr.utls
    called by "_oriented_minimum_bounding_box"

    (p1,p2) represents a line, not a segments
    input points are numpy arrays or lists
    '''
    (x0,y0), (x1,y1), (x2,y2) = p, p1, p2
    dx, dy = x2-x1, y2-y1
    return np.abs(dy*x0 -dx*y0 -x1*y2 +x2*y1) / np.sqrt(dx**2+dy**2)

################################################################################
def _linesIntersectionPoint(P1,P2, P3,P4):
    '''
    move to arr.utls
    called by "_oriented_minimum_bounding_box"

    line1 = P1,P2
    line2 = P3,P4

    returns the intersection "Point" of the two lines
    returns "None" if lines are parallel
    This function treats the line as infinit lines
    '''
    denom = (P1[0]-P2[0])*(P3[1]-P4[1]) - (P1[1]-P2[1])*(P3[0]-P4[0])
    if np.abs(denom) > np.spacing(1):
        num_x = ((P1[0]*P2[1])-(P1[1]*P2[0]))*(P3[0]-P4[0]) - (P1[0]-P2[0])*((P3[0]*P4[1])-(P3[1]*P4[0]))
        num_y = ((P1[0]*P2[1])-(P1[1]*P2[0]))*(P3[1]-P4[1]) - (P1[1]-P2[1])*((P3[0]*P4[1])-(P3[1]*P4[0]))
        return np.array([num_x/denom , num_y/denom])
    else:
        return None

################################################################################
def _convexHullArea(vertices):
    '''
    move to arr.utls

    called by "_oriented_minimum_bounding_box"

    vertices passed to this function in the form of numpy array
    '''
    ### sorting vertices CCW
    center = np.mean(vertices,axis=0)
    angletoVertex = [np.arctan2(p[1]-center[1] , p[0]-center[0])
                     for p in vertices]
    v = np.array([v for (t,v) in sorted(zip(angletoVertex,vertices))])

    ### calculatig the area, as a sum of triangles
    area = [np.cross([v[i,0]-v[0,0],v[i,1]-v[0,1]] , [v[i+1,0]-v[0,0],v[i+1,1]-v[0,1]])
            for i in range(1,len(v)-1)]
    area = 0.5 * np.sum(np.abs(area))

    return area

################################################################################
def _oriented_minimum_bounding_box (points):
    '''
    move to arr.utls

    arbitrary oriented minimum bounding box based on rotating caliper
    1_ for each simplex[i] in the polygon (segment between two consequtive vertices):
    a) line1 = simplex[i]
    b) pTemp = the fartherest vertex to line1
    c) line2 = parallel to line1, passing through pTemp
    d) caliper is the line1+line2 (two parallel line)
    e) lineTemp = perpendicular to the caliper's lines
    f) pTemp3 = the fartherest vertex to the lineTemp
    g) line3 = parallel to lineTemp, passing through pTemp3
    h) pTemp4 = the fartherest vertex to the line3
    i) line4 = parallel to line3, passing through pTemp4
    j) mbb according to simplex[i]: p1,p2,p3,p4 = intersection(line1,line2,line3,line4)
    k) calculate the area covered by (p1,p2,p3,p4)
    2_ among all the mbb created based on all simplices, pick the one with smallest area
    '''

    boundingBoxes = []
    area = []

    hull = scipy.spatial.ConvexHull(points)
    verticesIdx = list(set([vertex for simplex in hull.simplices for vertex in simplex ]))
    for simplex in hull.simplices:
        p1,p2 = hull.points[simplex]

        l12t = np.arctan2(p2[1]-p1[1] , p2[0]-p1[0])
        l34t = l12t+np.pi/2

        # line1 = p1,p2
        l1p1, l1p2 = p1, p2

        # line2 = construct the parallel from vIdx
        dis = list(np.abs([_distance2point(l1p1, l1p2, points[p]) for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l2p1, l2p2 = points[vIdx], points[vIdx]+np.array([np.cos(l12t),np.sin(l12t)])

        # lineTemp = arbitrary perpendicular line to line1 and line2.
        ltp1,ltp2 = p1, p1+np.array([np.cos(l34t),np.sin(l34t)])

        # line3 = parallel to lineTemp, passing through farthest vertex to lineTemp
        dis = list(np.abs([_distance2point(ltp1,ltp2, points[p])  for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l3p1, l3p2 = points[vIdx], points[vIdx]+np.array([np.cos(l34t),np.sin(l34t)])

        # line4 = parallel to line3, passing through farthest vertex to line3
        dis = list(np.abs([_distance2point(l3p1,l3p2, points[p])  for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l4p1, l4p2 = points[vIdx], points[vIdx]+np.array([np.cos(l34t),np.sin(l34t)])

        # BoundingBox = 4 points (intersections of line 1 to 4)
        vertices = np.array([ _linesIntersectionPoint(l1p1,l1p2, l3p1,l3p2),
                              _linesIntersectionPoint(l1p1,l1p2, l4p1,l4p2),
                              _linesIntersectionPoint(l2p1,l2p2, l3p1,l3p2),
                              _linesIntersectionPoint(l2p1,l2p2, l4p1,l4p2) ])

        # a proper convexhull algorithm should be able to sort these!
        # but my version of scipy does not have the vertices yet!
        # so I sort them by the angle
        center = np.mean(vertices,axis=0)
        angle2Vertex = [np.arctan2(p[1]-center[1] , p[0]-center[0]) for p in vertices]
        ### sorting bb according to tt
        BB = np.array([ v for (t,v) in sorted(zip(angle2Vertex,vertices)) ])
        boundingBoxes.append(BB)

        # find the area covered by the BoundingBox
        area.append(_convexHullArea(BB))

    # return the BoundingBox with smallest area
    val, idx = min((val, idx) for (idx, val) in enumerate(area))

    return boundingBoxes[idx]


################################################################################
################################################################################
################################################################################

def _create_mpath ( points ):
    '''
    move to arr.utls


    note: points must be in order
    - copied from mesh to OGM
    '''

    # start path
    verts = [ (points[0,0], points[0,1]) ]
    codes = [ mpath.Path.MOVETO ]

    # construct path - only lineto
    for point in points[1:,:]:
        verts += [ (point[0], point[1]) ]
        codes += [ mpath.Path.LINETO ]

    # close path
    verts += [ (points[0,0], points[0,1]) ]
    codes += [ mpath.Path.CLOSEPOLY ]

    # return path
    return mpath.Path(verts, codes)

################################################################################
def _get_pixels_in_mpath(path, image_shape=None):
    '''
    given a path and image_size, this method return all the pixels in the path
    that are inside the image
    '''

    # find the extent of the minimum bounding box, sunjected to image boundaries
    mbb = path.get_extents()
    if image_shape is not None:
        xmin = int( np.max ([mbb.xmin, 0]) )
        xmax = int( np.min ([mbb.xmax, image_shape[1]-1]) )
        ymin = int( np.max ([mbb.ymin, 0]) )
        ymax = int( np.min ([mbb.ymax, image_shape[0]-1]) )
    else:
        xmin = int( mbb.xmin )
        xmax = int( mbb.xmax )
        ymin = int( mbb.ymin )
        ymax = int( mbb.ymax )

    x, y = np.meshgrid( range(xmin,xmax+1), range(ymin,ymax+1) )
    mbb_pixels = np.stack( (x.flatten().T,y.flatten().T), axis=1)

    in_path = path.contains_points(mbb_pixels)

    return mbb_pixels[in_path, :]


################################################################################
################################################### edge neighborhood and pixels
################################################################################


################################################################################
def _pixel_neighborhood_of_segment (p1,p2, neighborhood=5):
    '''
    move to arr.utls

    Input:
    ------
    p1 and p2, the ending points of a line segment

    Parameters:
    -----------
    neighborhood:
    half the window size in pixels (default:5)

    Output:
    -------
    neighbors: (nx2) np.array


    Note:
    Internal naming convention for the coordinates order is:
    (x,y) for coordinates - (col,row) for image
    However, this method is not concerned with the order.
    If p1 and p2 are passed as (y,x)/(row,col) the output will
    follow the convention in inpnut.
    '''
    # creating a uniform distribution of points on the line
    # for small segments N will be zero, so N = max(2,N)
    N = int( np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) )
    N = np.max([2, N])
    x = np.linspace(p1[0],p2[0], N, endpoint=True)
    y = np.linspace(p1[1],p2[1], N, endpoint=True)
    line_pts = np.stack( (x.T,y.T), axis=1)

    # index to all points in the minimum bounding box (MBB)
    # (MBB of the line from p1 to p2) + margin
    xMin = np.min([ int(p1[0]), int(p2[0]) ]) - neighborhood
    xMax = np.max([ int(p1[0]), int(p2[0]) ]) + neighborhood
    yMin = np.min([ int(p1[1]), int(p2[1]) ]) - neighborhood
    yMax = np.max([ int(p1[1]), int(p2[1]) ]) + neighborhood
    x, y = np.meshgrid( range(xMin,xMax), range(yMin,yMax) )
    mbb_pixels = np.stack( (x.flatten().T,y.flatten().T), axis=1)

    # min distance between points in MBB and the line
    dists = scipy.spatial.distance.cdist(line_pts, mbb_pixels, 'euclidean')
    dists = dists.min(axis=0)

    # flagging all the points that are in the neighborhood
    # of the
    neighbors_idx = np.nonzero( dists<neighborhood )[0]
    neighbors = mbb_pixels[neighbors_idx]

    return neighbors

################################################################################
def _pixel_neighborhood_of_halfedge (arrangement, index_set,
                                    neighborhood=5, image_size=None):
    '''
    move to arr.arr

    Inputs:
    -------
    arrangement:
    index_set: an index-set to a half-edge

    Parameters:
    -----------
    neighborhood:
    half the window size in pixels (default:5)

    output:
    -------
    integer coordinate (index) to the half-edge's enighborhood.

    Note:
    -----
    output is in this format:
    (x,y) / (col,row)
    for platting use directly
    for image indexing use inverted
    '''
    s, e, k = index_set
    he = arrangement.graph[s][e][k]['obj']
    trait = arrangement.traits[he.traitIdx]
    pt_1 = arrangement.graph.node[s]['obj'].point
    pt_2 = arrangement.graph.node[e]['obj'].point

    if not( isinstance(trait.obj, (sym.Line, sym.Segment, sym.Ray) ) ):
        raise (StandardError(' only line trait are supported for now '))

    # Assuming only line segmnent - no arc-circle
    p1 = np.array([pt_1.x,pt_1.y]).astype(float)
    p2 = np.array([pt_2.x,pt_2.y]).astype(float)

    neighbors = _pixel_neighborhood_of_segment (p1,p2, neighborhood)

    if image_size is None:
        return neighbors
    else:
        xMin, yMin, xMax, yMax = [0,0, image_size[1], image_size[0]]
        x_in_bound = (xMin<=neighbors[:,0]) & (neighbors[:,0]<xMax)
        y_in_bound = (yMin<=neighbors[:,1]) & (neighbors[:,1]<yMax)
        pt_in_bound = x_in_bound & y_in_bound
        in_bound_idx = np.where(pt_in_bound)[0]
        return neighbors[in_bound_idx]



################################################################################
######################################################### connectivity map stuff
################################################################################
def _set_face_occupancy_attribute(arrangement, image, occupancy_threshold=200):
    '''
    '''
    for face in arrangement.decomposition.faces:
        pixels_in_path = _get_pixels_in_mpath(face.path, image_shape=image.shape)
        pixels_val = image[pixels_in_path[:,1],pixels_in_path[:,0]]
        total = float(pixels_val.shape[0])
        occupied = float(np.count_nonzero(pixels_val < occupancy_threshold))
        face.attributes['occupancy'] = [occupied, total]
    return arrangement

################################################################################ arrangement attr setting
def _set_face_centre_attribute(arrangement, source=['nodes','path'][0]):
    '''
    assumes all the faces in arrangement are convex

    if source == 'nodes' -> centre from nodes of arrangement
    if source == 'path' -> centre from vertrices of the face
    (for source == 'path', path must be up-todate)
    '''
    for face in arrangement.decomposition.faces:

        if source == 'nodes':
            nodes = [arrangement.graph.node[fn_idx]
                     for fn_idx in face.get_all_nodes_Idx()]
            xc = np.mean([ node['obj'].point.x for node in nodes ])
            yc = np.mean([ node['obj'].point.y for node in nodes ])
            face.attributes['centre'] = [float(xc),float(yc)]
        elif source == 'path':
            face.attributes['centre'] = np.mean(face.path.vertices[:-1,:], axis=0)

################################################################################
def _construct_connectivity_map(arrangement, set_coordinates=True):
    '''

    Parameter:
    set_coordinates: Boolean (default:True)
    assumes all the faces in arrangement are convex, and set the coordinate of
    each corresponding node in the connectivity-map to the center of gravity of
    the face's nodes

    Note
    ----
    The keys to nodes in connectivity_map corresponds to
    face indices in the arrangement
    '''

    connectivity_map = nx.MultiGraph()

    if set_coordinates: _set_face_centre_attribute(arrangement)
    faces = arrangement.decomposition.faces

    ########## node construction (one node per each face)
    nodes = [ [f_idx, {}] for f_idx,face in enumerate(faces) ]
    connectivity_map.add_nodes_from( nodes )

    # assuming convex faces, node coordinate = COG (face.nodes)
    if set_coordinates:
        for f_idx,face in enumerate(faces):
            connectivity_map.node[f_idx]['coordinate'] = face.attributes['centre']

    ########## edge construction (add if faces are neighbor and connected)
    corssed_halfedges = [ (s,e,k)
                          for (s,e,k) in arrangement.graph.edges(keys=True)
                          if arrangement.graph[s][e][k]['obj'].attributes['crossed'] ]

    # todo: detecting topologically distict connection between face
    # consider this:
    # a square with a non-tangent circle enclosed and a vetical line in middle
    # the square.substract(circle) region is split to two and they are connected
    # through two topologically distict paths. hence, the graph of connectivity
    # map must be multi. But if two faces are connected with different pairs of
    # half-edge that are adjacent, these connection pathes are not topologically
    # distict, hence they should be treated as one connection

    # todo: if the todo above is done, include it in dual_graph of the
    # arrangement

    # for every pair of faces an edge is added if
    # faces are neighbours and the shared-half_edges are crossed
    for (f1Idx,f2Idx) in itertools.combinations( range(len(faces)), 2):
        mutualHalfEdges = arrangement.decomposition.find_mutual_halfEdges(f1Idx, f2Idx)
        mutualHalfEdges = list( set(mutualHalfEdges).intersection(set(corssed_halfedges)) )
        if len(mutualHalfEdges) > 0:
            connectivity_map.add_edges_from( [ (f1Idx,f2Idx, {}) ] )

    return arrangement, connectivity_map


################################################################################ arrangement attr setting
def _set_edge_crossing_attribute(arrangement, skiz,
                                neighborhood=3, cross_thr=12):
    '''
    Parameters
    ----------
    neighborhood = 5 # the bigger, I think the more robust it is wrt skiz-noise
    cross_thr = 4 # seems a good guess! smaller also works

    Note
    ----
    skiz lines are usually about 3 pixel wide, so a proper edge crossing would
    result in about (2*neighborhood) * 3pixels ~ 6*neighborhood
    for safty, let's set it to 3*neighborhood

    for a an insight to its distributions:
    >>> plt.hist( [arrangement.graph[s][e][k]['obj'].attributes['skiz_crossing'][0]
                   for (s,e,k) in arrangement.graph.edges(keys=True)],
                  bins=30)
    >>> plt.show()

    Note
    ----
    since "_set_edge_occupancy" counts low_values as occupied,
    (invert=True) must be set when calling the _skiz_bitmap
    '''

    _set_edge_occupancy(arrangement,
                        skiz, occupancy_thr=127,
                        neighborhood=neighborhood,
                        attribute_key='skiz_crossing')


    for (s,e,k) in arrangement.graph.edges(keys=True):
        o, n = arrangement.graph[s][e][k]['obj'].attributes['skiz_crossing']
        arrangement.graph[s][e][k]['obj'].attributes['crossed'] = False if o <= cross_thr else True

    # since the outer part of arrangement is all one face (Null),
    # I'd rather not have any of the internal faces be connected with Null
    # that might mess up the structure/topology of the connectivity graph
    forbidden_edges  = arrangement.get_boundary_halfedges()
    for (s,e,k) in forbidden_edges:
        arrangement.graph[s][e][k]['obj'].attributes['crossed'] = False
        (ts,te,tk) = arrangement.graph[s][e][k]['obj'].twinIdx
        arrangement.graph[ts][te][tk]['obj'].attributes['crossed'] = False
    # return skiz


################################################################################ arrangement attr setting
def _set_edge_occupancy(arrangement,
                       image, occupancy_thr=200,
                       neighborhood=10,
                       attribute_key='occupancy'):
    '''
    This method sets the occupancy every edge in the arrangement wrt image:
    arrangement.graph[s][e][k]['obj'].attributes[attribute_key] = [occupied, neighborhood_area]

    Inputs
    ------
    arrangement:
    The arrangement corresponding to the input image

    image: bitmap (gray scale)
    The image represensts the occupancy map and has high value for open space.


    Parameters
    ----------
    occupancy_thr: default:200
    Any pixel with value below "occupancy_thr" is considered occupied.

    neighborhood: default=10
    half the window size (ie disk radius) that defines the neighnorhood

    attribute_key: default:'occupancy'
    The key to attribute dictionary of the edges to store the 'occupancy'
    This method is used for measuring occupancy of edges against occupancy map and skiz_map
    Therefor it is important to store the result in the atrribute dictionary with proper key


    Note
    ----
    "neighborhood_area" is dependant on the "neighborhood" parameter and the length of the edge.
    Hence it is the different from edge to edge.
    '''
    for (s,e,k) in arrangement.graph.edges(keys=True):
        index_set = (s,e,k)
        neighbors = _pixel_neighborhood_of_halfedge (arrangement, index_set,
                                                     neighborhood,
                                                     image_size=image.shape)

        neighbors_val = image[neighbors[:,1], neighbors[:,0]]
        occupied = np.nonzero(neighbors_val<occupancy_thr)[0]

        # if neighbors.shape[0] is zero, I will face division by zero
        # when checking for occupancy ratio
        o = occupied.shape[0]
        n = np.max([1,neighbors.shape[0]])

        arrangement.graph[s][e][k]['obj'].attributes[attribute_key] = [o, n]