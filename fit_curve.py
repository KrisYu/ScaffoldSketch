import numpy as np
from scipy import interpolate
from scipy.optimize import minimize


M = np.asfarray([
    [ -1, 3, -3, 1 ],
    [ 3, -6, 3, 0 ],
    [ -3, 3, 0, 0 ],
    [ 1, 0, 0, 0 ]])


## Helpers
def resample( points, inc = 0.01 ):
    '''
    Given: 
        points: [P0, P1, ..., Pn-1] raw points
    Returns:
        resampled points, 0.01 cm per point
    '''
    points = np.asarray(points)

    n_points, dim = points.shape

    # Parametrization parameter s.
    dp = np.diff(points, axis=0)                 # difference between points
    dp = np.linalg.norm(dp, axis=1)              # distance between points
    d = np.cumsum(dp)                            # cumsum along the segments
    d = np.hstack([[0],d])                       # add distance from first point
    length = d[-1]                               # length of point sequence
    num = int(length/inc) +1                     # number of samples
    s = np.linspace(0,length,num)                # sample parameter and step


    # Compute samples per dimension separately.
    q = np.zeros([num, dim])
    for i in range(dim):
        q[:,i] = np.interp(s, d, points[:, i])
    return q

def closest_point_from_list(q, points):
    '''
    Given:
        q: a point
        points: [P0, P1, ...]
    Return:
        p in points that most close to q
    '''

    index = None
    dist  = None
    point = None

    for i, p in enumerate(points):
        if dist is None or np.linalg.norm(q - p) < dist:
            dist = np.linalg.norm(q - p)
            index = i
            point = p.copy()
    return (index, point)

def fem_tangent(points):
    # use forward finite difference to calculate tangency at discrete points
    t = []
    for i in range(len(points) -1 ):
        t.append( (points[i+1] - points[i]) / np.linalg.norm(points[i+1] - points[i]) )
    # the last point
    t.append( t[-1] )
    return t

## Optimize Helpers
def bezier_curve( ts, c0, c1, c2, c3 ):
    '''
    Given:
        ts: a sequence of t values to evaluate the spline
        c0, c1, c2, c3: d-dimensional values specifying the four control points of the spline.
    Returns:
        points: Returns a sequence of d-dimensional points, each of which is the cubic spline evaluated at each t in `ts`.
            Returns the sequence as a `len(ts)` by d numpy.array.
    '''
    
    P = np.zeros( ( 4, len( c0 ) ) )
    P[0] = c0
    P[1] = c1
    P[2] = c2
    P[3] = c3
    Ts = np.tile( ts, ( 4, 1 ) ).T
    # print(np.asfarray(ts).shape)
    # print(Ts.shape)
    Ts[:,0] **= 3
    Ts[:,1] **= 2
    Ts[:,2] **= 1
    Ts[:,3] = 1
    return Ts @ (M @ P)

def evaluate(X, points, tangents, resolution = 0.01):
    '''
    Given:
        X : freedom, the magnitude of tangents
        points : 
        tangents:
    '''
    points = np.asarray(points)
    tangents = np.asarray(tangents)
    nPoints, dim = points.shape
    
    spline_points = np.empty((0,dim))

    k = 0
    
    for i in range(nPoints - 1):
        c0 = points[i]
        c3 = points[i+1]
        c1 = c0 + tangents[i] * X[k]
        c2 = c3 - tangents[i+1] * X[k+1]
        
        k = k + 2
        
        d = np.linalg.norm(c3-c0)
        
        # max(2) because we always want to sample at least once in the middle
        ts = np.linspace(0, 1, max( 2, int(d/resolution) ), endpoint= False)
        # add endpoint for the last segment
        if i == nPoints - 2:
            ## Add 1 for the second endpoint
            ts = np.linspace(0, 1, max( 2, int(d/resolution) ) + 1, endpoint=True)
        
        curve_points = bezier_curve(ts, c0, c1, c2, c3)
        spline_points = np.concatenate((spline_points, curve_points))
        

    return spline_points

def init_X(points):
    '''
    Given:
        points
    Return:
        X
    '''
    # init X 
    X  = np.ones(len(points) * 2 - 2) 
    
    # init X value to avoid self intersections
    k = 0
    for i in range(len(points)-1):
        p0 = points[i]
        p1 = points[i+1]
        d = np.linalg.norm(p1-p0)
        X[k] = X[k+1] = d/3
        k = k + 2
    
    return X

def menger_curvature(x, y, z):
    '''
    https://en.wikipedia.org/wiki/Menger_curvature
    Given:
        x, y, z : 3 points
    Return:
        menger curvature    
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    a = np.linalg.norm(x - y)
    b = np.linalg.norm(y - z)
    c = np.linalg.norm(z - x)
    
    area = np.linalg.norm( np.cross(x - y, z - x) )
    
    if area < 1e-20: result = 0
    else: result = 2 * area / ( a * b * c)
        
    return result

def menger_curvature_everywhere(pts):
    '''
    Given:
        pts: A polyline of 3D points
    Returns:
        The menger curvature at pts[1], pts[2], ..., pts[-2]
    '''
    
    pts = np.asarray(pts)
    ## This only works in 2D or 3D
    assert pts.shape[1] in (2,3)
    
    xy = pts[:-2] - pts[1:-1]
    yz = pts[1:-1] - pts[2:]
    zx = pts[2:] - pts[:-2]
    
    a = ( xy**2 ).sum( 1 )
    b = ( yz**2 ).sum( 1 )
    c = ( zx**2 ).sum( 1 )
    
    area = ( np.cross( xy, zx ) ** 2 )
    ## In 2D, the cross product function returns the z component, which is the only non-zero element.
    if pts.shape[1] == 3: area = area.sum(1)
    result = 2 * np.sqrt( area / ( a * b * c) )
    result[ area < 1e-20 ] = 0.
    
    return result

def edge_weights(pts):
    '''
    Given:
        pts: A polyline of 3D points
    Returns:
        The mass weighting factor for each edge in `pts`.
        The result has length one less than `pts`.
    '''
    
    pts = np.asarray(pts)
    
    edges = pts[:-1] - pts[1:]
    edge_lengths = np.sqrt( ( edges**2 ).sum(1) )
    return edge_lengths

def property_along_curve( samples ):
    '''
    Given:
        samples: An array of many points
    Returns:
        variation_of_curvature: An array of variation-of-curvature values computed along the curve. It will have length one smaller.
    '''

    curvature = menger_curvature_everywhere( samples )
    return curvature[1:] - curvature[:-1]

def loss( v ):
    return v**2
    
##  called by to generate shape line from keypoints
def shape_line_from_keypoints( state, curve, thresholds ):
    '''
    Given:
        state: program state, scaffolds information inside
        curve: curve pts from data
        thresholds: thresholds used to fit curve
    Return:
        optimized curve
    '''
    ### 1. Preprocessing. Remove duplicated points or very close points(< 1cm) from raw_curve_points, get resampled_curve_points.
    ### 2. Fit a line using PCA and see how the resampled_curve_points diverge with the line, decide to fit a line or curve with the input.
    ### 3. Find closest scaffold points with points along the curve. Also try to decide the tangent direction for that curve point.
    ###    Tangent direction: dir = (next - cur) except for the last point 
    ###    If there are line_direction parallel/perpendicular to that point direction, remember the direction.
    ### 
    ### 4 If the curve should be a straigt_line, use PCA to fit a straigt_line, otherwise use optimization find the minimal curvature varation curve.
    ### 5 Return it.


    ### 1. resmaple  
    # points from original curve
    xi = [ pt['x'] for pt in curve]
    yi = [ pt['y'] for pt in curve]
    zi = [ pt['z'] for pt in curve]
    
    curve_points = np.zeros([ len(curve) , 3])

    curve_points[:, 0] = xi
    curve_points[:, 1] = yi
    curve_points[:, 2] = zi
    
    # print('raw_curve_points',curve_points)

    # too few raw points, almost impossible
    if len(curve_points) < 2 : return 

    # resampled 1cm sample points
    curve_points = resample(curve_points)
    # too few resampled points 
    if len(curve_points) < 3 : return 
    # print('resampled_curve_points',curve_points)

    state['raw_shape_curves'].append( curve_points )
    # smoothed curve points
    # I don't want to smooth and see what happens
    # curve_points = smooth(curve_points)
    # too few smoothed points
    # if len(curve_points) < 3 : return 

    # add the smoothed raw curve points to debug
    # state['raw_shape_curves'].append( curve_points ) 

    # tangent direction of points
    # we need at least 2 points and this confirmed by previous if
    point_directions = fem_tangent( curve_points )

    ### 2. line or curve?
    curve_should_be_line =  straight_line_or_curve( curve_points )


    ### 3
    # scaffold points: endpoints and midpoints of scaffold lines
    scaffold_points = state['line_points']

    # too few scaffold points
    if len(scaffold_points) < 2: return 


    closest_scaffold_point_indices = []
    possible_dirs_at_key_points = {}

    # loop through the curve_points, find the closest_scaffold_point_indices - might be duplicated
    # find the possible directions near the scaffold points - key is index of scaffold points, value is the possible directions
    for curve_point_idx, curve_point in enumerate(curve_points):
        # find the point on scaffold that nearest to the point on resampled curve
        scaffold_point_idx, scaffold_point = closest_point_from_list( curve_point, scaffold_points )
        
        # Skip points too far away.
        if np.linalg.norm( curve_point - scaffold_point ) > thresholds['point_point_distance']:
            continue
        
        # scaffold_point_idx 
        closest_scaffold_point_indices.append( scaffold_point_idx )

        # record possible directions at scaffold key points 
        if scaffold_point_idx not in possible_dirs_at_key_points:
            possible_dirs_at_key_points[scaffold_point_idx] = []

        possible_dirs_at_key_points[scaffold_point_idx].append( point_directions[curve_point_idx] )


    # key indices - remove duplicate index
    # key_points = [ scaffold_points[i].tolist() for i in key_indices ]
    key_indices = []
    # key_directions - key: index of scaffold points, value - direction at that point 
    key_directions = {}

    # remove duplicate and add then to key_indices
    for index in closest_scaffold_point_indices:
        if key_indices == [] or index != key_indices[-1]:
            key_indices.append( index )

    # too few scaffold points, maybe here I need to pass the curve_points and curve_tangents to fit
    if len(key_indices) < 2: return 


    for index, directions in possible_dirs_at_key_points.items():
        # here the index is the index of the scaffolding index 
        # it is not the index in key_points
        # print( index, directions)
        point_most_possible_dir = None
        point_dir_scaffold_dir_dot_product = None 

        for point_dir in directions:
            # all the possible scaffold_dir at that point
            for scaffold_dir in state['points_info'][index]:
                # most parallel direction 
                if point_dir_scaffold_dir_dot_product is None or abs( 1- abs( np.dot( scaffold_dir, point_dir)) ) < point_dir_scaffold_dir_dot_product:
                    point_dir_scaffold_dir_dot_product = abs( 1 - abs( np.dot( scaffold_dir, point_dir)) )
                    # make sure its the right direction, but not the negative direction
                    if np.dot(scaffold_dir, point_dir) > 0:
                        point_most_possible_dir = scaffold_dir.copy()
                    else:
                        point_most_possible_dir = -scaffold_dir.copy()
        
        # check the direction within threshold
        if point_dir_scaffold_dir_dot_product < thresholds['same_direction_threshold']:            
            key_directions[index] = point_most_possible_dir
        else:
            # otherwise use the average of the directions near the point
            mean_direction_near_point = np.mean(directions, axis = 0) 
            # normalize
            mean_direction_near_point /= np.linalg.norm( mean_direction_near_point )

            key_directions[index] = mean_direction_near_point  



    
    key_tangents = []
    for index in key_indices:
        key_tangents.append( key_directions[index] )

    key_points = [ scaffold_points[i] for i in key_indices ]



    # print('###############################')
    # print('key_indices', key_indices) 
    # print('key_directions', key_directions)
    # print('scaffold_directions', scaffold_directions)
    # print('state = ', state)
    # print('curve_points = ', curve_points)
    # print('possible_dirs_at_scaffold_points = ', possible_dirs_at_key_points)
    # print('key_points = ', key_points)
    # print('key_tangents =', key_tangents)
    # print('###############################')
    if curve_should_be_line:
        # straight line only need 2 points 
        print('straight_line')
        # shape_points = resample( key_points )
        # print('key_points', np.asarray(key_points).tolist() )
        return np.asarray(key_points).tolist()
    else:
        print('curve')
        optimized_X = MVC_magnitudes(key_points, key_tangents)
        spline_points = evaluate( optimized_X, key_points, key_tangents)
        shape_points = resample( spline_points )

    # print('###############################')
    # print('sample', shape_points.tolist()) 
    # print('###############################')
    return shape_points.tolist()

## Fit Line
def straight_line_or_curve( curve_points ):
    '''
    Given:
        curve_points
    Return:
        True -> straight_line
        False -> curve 

    fit curve_points to straight_line
    check the diverge between curve_points and straight_line
    decide whether this should be straight_line or use later optimization way to fit curve
    '''

    line = pca_line( curve_points )
    line_length = np.linalg.norm( line[0] - line[1] )
    line_points = resample( line )

    # print('line_length', line_length )
    # print('curve_points count', len(curve_points))
    # print('line_points count', len(line_points))

    # line_points <= curve_porints
    n = len(line_points)
    n_diff = len(curve_points) - len(line_points)
    # max distance between 
    diffs = np.linalg.norm(curve_points[:n] - line_points[:n], axis = 1)
    diff_ratio = np.amax(diffs) / line_length
    

    # print('diff_ratio', diff_ratio)
    
    # This should be a straight_line, if there are not many new points introduced 
    # and the deviation is not big
    # since this is relatively to the line and it use percentile
    # this should not be controlled by the thresholds


    # n_diff : how many new points introduced
    # diff_ratio : deviation to line
    # print('n_diff', n_diff)
    # print('diff_ratio', diff_ratio)
    # based on the user study results, the parameters could be bigger
    if n_diff <= int( 0.15 * len(curve_points) ) + 1 and diff_ratio < 0.1:
        return True
    return False
   

def pca_line( points ):
    '''
    '''

    ### We'll fit a line by PCA.
    ### 1 Center the points.
    ### 2 Project the centered points along the first axis.
    ### 3 Take the min and max projections as the endpoints.
    ### 4 Rotate back.
    
    ### 1
    center = np.mean( points, axis = 0 )
    points = points - center
    
    ### 2
    _, _, Vt = np.linalg.svd( points )
    curve_rot = Vt[0,:] @ points.T
    
    ### 3
    first = curve_rot.min()
    last = curve_rot.max()
    
    ### 4
    segment_unrot = np.asfarray( [ Vt[0,:] * first, Vt[0,:] * last ] )
    segment_unrot += center
        
    return segment_unrot   

## Optimizer
def MVC_magnitudes(points, tangents, original_curve_pts = None ):
    '''
    Given:
        num: how many points we get from the spline to calculate the change of curvature
        points: A sequence of N d-dimensional points to interpolate
        tangents: A sequence of d-dimensional tangent directions at each point
        original_curve_pts: (optional) A sequence of N sequences of data points between the start and end point of each cubic segment.
    Returns:
        hermite: A sequence of N-1 cubic Hermite segments. Segment i
                 interpolates the points `ptsi]` and `ptsi+1]` with corresponding
                 tangent directions `tangentsi]` and `tangentsi+1]`.
                 The magnitudes of `tangents` are ignored.
    '''
    
    
    ### We want to optimize a property of the spline constructed
    ### from the sequence of points and tangents. The degrees of freedom
    ### are the magnitudes of those tangents.
    ### We need a function to evaluate the property we want to measure.
    ### The property in general can be positive or negative,
    ### We want to minimize some "loss" function of the property,
    ### like the absolute value, or the square, or a fancier function.
    ### In our case, we want to measure (and minimize) the variation in curvature.
    ### We could try for an analytic solution to all or part of this problem by,
    ### for example, computing the integral of the squared variation of curvature
    ### directly and then solving for the derivative equal to 0 or
    ### solving the Euler-Lagrange equations.
    ### In general, it will be difficult to do this, so we can approximate this
    ### by sampling everything.
    ### That means, we need:
    ### 1) a function to sample each piecewise curve: sample()
    ### 2) a function to compute the property we want to measure along the sampled curve (e.g. given a sample point and some of its neighbors): property( samplei-1 samplei samplei+1] )
    ### 3) a loss function: loss
    ### We also need a function to create piecewise curves given the current degrees-of-freedom: unpack()
    
    points = np.asfarray( points )
    tangents = np.asfarray( tangents )
    
    assert(len(points) == len(tangents))
    assert(len(points) >= 2)
    
    X = init_X(points)

    
    def f(X):
        evaluated_bezier_points = evaluate(X, points, tangents)
        prop = property_along_curve( evaluated_bezier_points )
        
        ## unweighted
        # E = sum( [loss( v ) for v in prop ])
        # E = loss(prop).sum()
        
        ## weighted
        E = ( loss(prop)*edge_weights( evaluated_bezier_points )[1:-1] ).sum()
        
        return E
    
    result = minimize( f, X, method = 'BFGS', jac = None, tol = 0.0001, options = { 'disp': False, 'eps': 0.000001, 'gtol': 0.0001, 'maxiter': 1000 } )
    # print(result)


    # I do not want use the optimized result if the result.x is far from original
    # 3 might create a cusp or self-intersection
    if any( np.abs(result.x/X) > 3):
        return X

    return result.x

