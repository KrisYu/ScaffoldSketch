from __future__ import print_function, division
from os import stat

import numpy as np


import fit_line
import fit_point
import fit_curve


'''
This module stores the state for the scaffold sketching program.
The state is a dictionary with certain keys.
Call `make_new_program_state()` to create one.
'''

def make_new_program_state():
    '''
    Returns a dictionary for storing the state of a scaffold sketching
    program. 
    
    The dictionary has keys value pairs

    'construction_lines' : [ list of ndarray as construction lines ]
    'shape_curves' :  [ list of points ]
    'raw_construction_lines' : [ list of tuple as raw points]
    'raw_shape_curves' :  [ list of ndarray as raw shape points ]
    'line_points': [ np.array([x,y,z]) ]  points attach to lines
    'line_directions' : [ np.array([0, 1, 0]) ]  # start with vertical direction
    'line_lengths': [ float as line_length ]
    'points_info': [ points_idx: [ a list of ndarray as direction at that point] ]
    '''
    state = { 'construction_lines' : [],
              'shape_curves' : [], 
              'raw_construction_lines': [], # for debug
              'raw_shape_curves' : [], # for debug
              'line_points' : [], 
              'line_directions' : [ np.array([0, 1, 0]) ], 
              'line_lengths' : [],
              'points_info' : {}
            }

    return state


def make_line_thresholds( scale ):
    '''
    init fit_line thresholds with the scale factor
    '''
    thresholds = {
    'point_point_distance2': 0.05 ** 2, # 0.05 **2 
    'point_line_distance2': 0.05 ** 2, # 0.05 **2 
    'line_vec_parallel2': (1 - np.cos(10/180*np.pi)) ** 2, # 0.015 **2
    'line_vec_perpendicular2': np.cos(80/180*np.pi) ** 2, # 0.17 ** 2
    'line_length_ratio2': 0.25 ** 2 # 75% ~125%?
    }
    
    thresholds['point_point_distance2'] /= scale
    thresholds['point_line_distance2'] /= scale
    
    return thresholds



def line_or_point(state, data ):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        data: from html side, a dictionary, have 'scale' and 'pts', example below:
        {'scale': {'x': 1, 'y': 1, 'z': 1}, 
         'pts': [{'x': 0, 'y': 0, 'z': 0},
                 {'x': 0, 'y': 1, 'z': 0}]}
        
    Returns:
        use this function to decide whether it's a line or point
    '''
    ### Use the scale to set up thresholds
    scale = data['scale'] 
    pts = data['pts']

    # print('scale', scale)
    # print('data', data)

    thresholds = make_line_thresholds( scale['x'] )
    

    ### 1 Get startpoint and endpoint of line
    ### 2 Depend on how many constraints, how long the line is, let it be a point or line

    x0 = pts[0]['x']
    y0 = pts[0]['y']
    z0 = pts[0]['z']

    x1 = pts[1]['x']
    y1 = pts[1]['y']
    z1 = pts[1]['z']

    startpoint = (x0, y0, z0) 
    endpoint = (x1, y1, z1) 

    # print('startpoint', startpoint)
    # print('endpoint', endpoint)
    # print('start_end_dist', start_end_dist)
    
    constraints = fit_line.build_energy_for_line_segment(startpoint, endpoint, thresholds, state)
    start_end_dist = np.linalg.norm( np.asarray(startpoint) - np.asarray(endpoint) )

    # less than 4 constraints
    # The point is just projection, so we don't need scale factor
    if start_end_dist < 0.03: # less than 0.03, by no means make it a point?
        result = incorporate_new_raw_point( state, pts)
        print('point')
    elif len(constraints) <= 4 and start_end_dist < 0.08: # 8cm
        result = incorporate_new_raw_point( state, pts )
        print('point')
    else:
        result = incorporate_new_raw_construction_line( state, ( startpoint, endpoint), thresholds )
        print('line')

    return result


def incorporate_new_raw_construction_line( state, line , thresholds):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        raw_line: startpoint and endpoint of raw construction line
        thresholds: current thresholds under scale factor
    Returns:
        line: a part of points which are the start and end of the new construction line
    
    Modified `state` to add a new construction line based off
    of the raw GUI input `pts`.

    Called by line_or_point function.
    '''
    
    ### 1 Store the raw construction line to state.
    ### 2 Snap the fit line to existing construction lines.
    ### 3 Add line, line direction, points, length to state if not current exist.
    
    ### 1
    state['raw_construction_lines'].append( line )

    ### 2
    snapped_line = fit_line.snap_line_to_other_lines( line, state, thresholds )

    ### 3
    fit_line.add_new_line_info( snapped_line, state )
    
    return snapped_line

def incorporate_new_raw_point( state, pts ):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        pts: raw points from the GUI as a sequence of (x,y,z?) triplets.
    Returns:
        point
    
    Modified `state` to add a new point based off
    of the raw GUI input `pts`.

    Called by line_or_point function.
    '''

    # find the center of pts, project it to the nearest line
    point = fit_point.point_fitting(state, pts)
    return point 


def make_curve_thresholds( scale ):
    '''
    init fit_curve thresholds with the scale factor
    '''

    thresholds = {
        'point_point_distance' : 0.03, # 3cm
        'same_direction_threshold' : (1 - np.cos(25/180*np.pi)) # 25 degree
    }

    thresholds['point_point_distance'] /= scale

    return thresholds

    
def incorporate_new_raw_shape_line( state, data ):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        data: from html side
        'scale' : scale vector
        'pts': pts
    
    Returns:
        curve: curve points
    
    Modified `state` to add a new construction line based off
    of the raw GUI input `pts`.
    '''

    ### Use the scale to set up thresholds
    scale = data['scale'] 
    pts = data['pts']

    thresholds = make_curve_thresholds( scale['x'] )

    ### 1 Fit a curve to the points.
    ### 2 Store all curve points in state.

    curve = fit_curve.shape_line_from_keypoints( state, pts, thresholds )
    if curve:
        state['shape_curves'].append( curve )
        return curve


def prepare_state_for_UI( state ):
    """
    construction_lines: list of ndarray
    shape_lines: list of ndarray
    
    prepare them to json so it is easier for javascript to parse
    
    used for undo and redo
    to display, html/js only need to know lines, points and curves
    """
    current_state = { 'lines' : [],  'points': [], 'curves' : [] }

    for construction_line in state['construction_lines']:
        current_state['lines'].append( construction_line.tolist() )
    
    for point in state['line_points']:
        current_state['points'].append( point.tolist() )
    
    for shape_curve in state['shape_curves']:
        current_state['curves'].append( shape_curve )

    return current_state


def prepare_state_for_save_all_info( state ):
    '''
    Given:
        state
    Return:
        serializable of state
    

    Object of type ndarray is not JSON serializable
    Used to save all information 
    '''
     
    current_state = {  'construction_lines' : [], # list of ndarray
                       'shape_curves' : [], # list of list
                       'raw_construction_lines': [], # list of ndarray
                       'raw_shape_curves': [],
                       'line_points' : [], 
                       'line_directions': [], # list of ndarray
                       'line_lengths': [],
                       'points_info': {}
                    }


    for construction_line in state['construction_lines']:
        current_state['construction_lines'].append( construction_line.tolist() )

    for shape_curve in state['shape_curves']:
        current_state['shape_curves'].append( shape_curve )

    for raw_construction_line in state['raw_construction_lines']:
        current_state['raw_construction_lines'].append( raw_construction_line )
    
    for raw_shape_curve in state['raw_shape_curves']:
        current_state['raw_shape_curves'].append( raw_shape_curve.tolist() )

    for point in state['line_points']:
        current_state['line_points'].append( point.tolist() )

    for dir in state['line_directions']:
        current_state['line_directions'].append( dir.tolist() )

    for length in state['line_lengths']:
        current_state['line_lengths'].append( length )

    for point, infos in state['points_info'].items():
        current_state['points_info'][point] = []
        for info in infos:
            current_state['points_info'][point].append( info.tolist() )

    return current_state


def load_state_from_json( state ):
    '''
    opposite of prepare_state_for_save_all_info
    '''


    current_state = {  'construction_lines' : [], # list of ndarray
                       'shape_curves' : [], # list of list
                       'raw_construction_lines': [], # list of ndarray
                       'raw_shape_curves': [],
                       'line_points' : [], 
                       'line_directions': [], # list of ndarray
                       'line_lengths': [],
                       'points_info': {}
                    }
    for construction_line in state['construction_lines']:
        current_state['construction_lines'].append( np.asarray(construction_line) )

    for raw_construction_line in state['raw_construction_lines']:
        current_state['raw_construction_lines'].append( raw_construction_line )

    for shape_curve in state['shape_curves']:
        current_state['shape_curves'].append( shape_curve )
 
    for raw_shape_curve in state['raw_shape_curves']:
        current_state['raw_shape_curves'].append( np.asarray(raw_shape_curve) )

    for point in state['line_points']:
        current_state['line_points'].append( np.asarray(point) )
        
    for dir in state['line_directions']:
        current_state['line_directions'].append( np.asarray(dir) )
        
    for length in state['line_lengths']:
        current_state['line_lengths'].append( length )

    for point, infos in state['points_info'].items():
        current_state['points_info'][int(point)] = []
        for info in infos:
            current_state['points_info'][int(point)].append( np.asarray(info) )
            
    return current_state