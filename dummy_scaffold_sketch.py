from __future__ import print_function, division

import numpy as np


import fit_line
import fit_point
import fit_curve


'''
This is just a dummpy scaffold sketch.
It will just return the raw data.
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


def line_or_point(state, data ):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        data: from html side
        'scale' : scale vector
        'pts': pts
        
    Returns:
        use this function to decide whether it's a line or point
    '''
    ### Use the scale to set up thresholds
    scale = data['scale'] 
    pts = data['pts']

    # print('scale', scale)
    # print('data', data)

    

    ### 1 Get startpoint and endpoint of line
    ### 2 Depend on how many constraints, how long the line is, let it be a point or line
    
    startpoint, endpoint = fit_line.endpoints_line( pts )
    startpoint = np.asarray( startpoint )
    endpoint = np.asarray( endpoint )
    

    if np.linalg.norm( startpoint - endpoint ) < 0.03: # less than 0.03, by no means make it a point?
        result = incorporate_new_raw_point( state, pts)
        print('point')
    else:
        result = incorporate_new_raw_construction_line( state, pts )
        print('line')

    return result

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
    point = fit_point.point_fitting(state, pts)
    return point 

def incorporate_new_raw_construction_line( state, pts ):
    '''
    Given:
        state: a state as returned by `make_new_program_state`
        pts: startpoint and endpoint of construction line
    Returns:
        line: a part of points which are the start and end of the new construction line
    
    Modified `state` to add a new construction line based off
    of the raw GUI input `pts`.
    '''

    line = fit_line.endpoints_line( pts )
    line = np.asarray(line)

    state['construction_lines'].append( line )

    return line


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

    ### 1 Fit a curve to the points.
    ### 2 Store all curve points in state.

    xi = [ pt['x'] for pt in pts]
    yi = [ pt['y'] for pt in pts]
    zi = [ pt['z'] for pt in pts]
    
    curve_points = np.zeros([ len(pts) , 3])

    curve_points[:, 0] = xi
    curve_points[:, 1] = yi
    curve_points[:, 2] = zi

    # print( curve_points )

    points = curve_points.tolist()

    state['shape_curves'].append( points )
    return points



def prepare_state_for_UI( state ):
    """
    construction_lines: list of ndarray
    shape_lines: list of ndarray
    
    prepare them to json so it is easier for javascript to parse
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
    This is save all info 
    '''

    """
     construction_lines: list of ndarray
     shape_lines: list of ndarray
     
     Object of type ndarray is not JSON serializable

     convert things to list 
     """
     
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


