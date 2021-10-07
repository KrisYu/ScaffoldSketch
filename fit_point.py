import autograd.numpy as np
import math_tool as mt

def point_fitting(state, pts):
    '''
    Simple way to fit point
        find the center of pts, project it to the nearest line
    '''


    ### 1 Find the centroid of pts.
    ### 2 Find the closest line with the centroid.
    ### 3 Project the point onto the line.
    ### 4 If point exists, add the point and related info to state.
    ### There's issue possibility - no line, no point.


    ### 1
    xi = [ pt['x'] for pt in pts]
    yi = [ pt['y'] for pt in pts]
    zi = [ pt['z'] for pt in pts]

    points = np.zeros([ len(pts) , 3])

    points[:, 0] = xi
    points[:, 1] = yi
    points[:, 2] = zi    
    
    centroid = np.mean(points, axis=0)


    ### 2 
    closest_line = None
    closest_dist = None

    for line in state['construction_lines']:
        dist = mt.point_line_distance2( centroid, line )
        if closest_dist is None or dist < closest_dist:
            closest_line = line 
            closest_dist = dist
    
    ### 3
    point = None
    # draw the point on this line
    if not (closest_line is None):
        point = mt.closest_point_on_line( centroid, closest_line )
    
    ### 4
    # add point to line_points and related info to point
    if not (point is None):
        add_new_point_info(point, closest_line, state)
    
    return point

def add_new_point_info( point, line, state ):
    """
    Given:
        point: 
        line: np.array( [start, end]) 
        state: current scaffold information
    Return:
        no return, add the point and related tangent info to state
    """
    p0, p1 = line[0], line[1]
    
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)

    line_dir = mt.dir( p1 - p0 ) # line direction

    mt.add_point_and_assosiated_directions(point, line_dir, state)

