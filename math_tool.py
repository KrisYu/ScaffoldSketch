import autograd.numpy as np
# import numpy as np

## point and line tools
# used in fit_line and fit_point
def mag2( v ):
    """
    Given:
        v : vec
    Return:
        l2 norm of vec
    """
    v = np.asarray( v )
    return np.dot(v, v)

def dir( v ):
    """
    Given:
        v : vec
    Return:
        normalized vec
    """
    l2norm = mag2( v )
    assert l2norm > 0
    return v / np.sqrt( l2norm )

def point_point_distance2( point1, point2 ):
    point1 = np.asarray( point1 )
    point2 = np.asarray( point2 )
    
    v = ( point1 - point2 )
    return  np.dot( v, v ) 

def point_point_distance( point1, point2 ):
    return np.sqrt( point_point_distance2( point1, point2) )

def point_line_distance2(point, line):
    # cross product = area of parallelogram, divided by length
    # height = area / length
    p, q = line
    p = np.asarray(p)
    q = np.asarray(q)
    point = np.asarray(point)

    return mag2( np.cross(p - q, p - point) )/ mag2(p - q)

def line_vec_parallel2( line , vec ):
    # 1 - np.abs( np.dot( lvec, np.asarray(vec) )) in (0, 0.015)
    lstart = np.asarray( line[0] )
    lend = np.asarray( line[1] )
    
    lvec = dir( lend - lstart )
    
    return ( 1 - np.abs( np.dot( lvec, np.asarray(vec) )) ) ** 2

def line_vec_perpendicular2( line, vec ):
    # np.dot( lvec, np.asarray(vec) ) in (0, 0.17)
    # cos x around 90 degree is more steep than cos x around 0
    lstart = np.asarray( line[0] )
    lend = np.asarray( line[1] )
    
    lvec = dir( lend - lstart )

    return ( np.dot( lvec, np.asarray(vec) )) ** 2

def line_length_ratio2( line, line_length ):

    lstart = np.asarray( line[0] )
    lend = np.asarray( line[1] )
    
    return ( point_point_distance( lstart, lend ) /line_length  - 1 ) ** 2
 
def closest_point_on_line(p, line):
    '''
    Given : line and point p not on the line
    Return : closest point on the line to point p
    '''
    # https://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line
    a, b = line
    ap = p - a
    ab = b - a
    return a + np.dot(ap,ab)/np.dot(ab,ab) * ab


# used in both fit_line and fit_point
def add_point_and_assosiated_directions( point, direction, state):
    '''
    Given:
        point:
        direction: point assosiated direction
        state: 
    Return:
        no return 
    '''

    ## No info of this point has been recorded
    ## We add both the point and related direction to state 

    # atol = 1e-4 betweeen points, wondering this could be larger?
    if not any(np.allclose(point, x, atol=1e-4) for x in state['line_points']):
        
        # add point related direction
        point_idx = len(state['line_points'])
        state['points_info'][point_idx] = [direction]
        
        # add point to line_points
        state['line_points'].append( point )
    
    ## Point exists
    ## Add the point related direction

    # point exists in state['line_points'], try add the assosiated direction
    else:
        # first, find point in point_info
        # second, make sure the line_dir not in point_info
        for point_idx, existing_point in enumerate(state['line_points']):
            if np.allclose(point, existing_point, atol = 1e-4):
                if not any(np.allclose(direction,x, atol=0.015) for x in state['points_info'][point_idx]) and \
                    not any(np.allclose(direction, -x, atol=0.015) for x in state['points_info'][point_idx]):
                    state['points_info'][point_idx].append( direction )


# might be used
def point_weights(pts):
    '''
    Given:
        pts: A polyline of 3D points
    Returns:
        The mass weighting factor for each point in pts
    
    
    Let pts = [ p0, p1, p2, p3 ]
    
    edges = [ p1-p0, p2-p1, p3-p2 ]
    
    weights = [ .5*|p1-p0|, .5*( |p1-p0| + |p2-p1| ), .5*( |p2-p1| + |p3-p2| ), .5*|p3-p2| ]
    aka
    weights = .5*[ |edges[0]|, |edges[0]|+|edges[1]|, |edges[1]|+|edges[2]|, |edges[2]| ]
    aka
    weights[:] = 0
    weights[:-1] += .5*|edges|
    weights[1:] += .5*|edges|
    '''
    
    pts = np.asarray(pts)
    
    edges = pts[:-1] - pts[1:]
    edge_lengths = np.sqrt( ( edges**2 ).sum(1) )
    
    weights = np.zeros( len(pts) )
    weights[:-1] += .5*edge_lengths
    weights[1:] += .5*edge_lengths
    
    return weights