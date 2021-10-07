from os import stat
import autograd.numpy as np
from autograd import grad

import autograd

# vjp stands for vector-Jacobian product
autograd.extend.defvjp(
    autograd.numpy.asarray,
    lambda ans, *args, **kw: lambda g: g
)

import math_tool as mt



## Optimization helpers
def pack( startpoint, endpoint ):
    '''
    Given:
        startpoint: A sequence containing the ( x,y,z ) of the start point.
        endpoint: A sequence containing the ( x,y,z ) of the end point.
    Returns:
        X: The numpy.ndarray of degrees-of-freedom used by the optimization function representing a 3D line segment.
    '''
    
    return np.concatenate( [ startpoint, endpoint ] )

def unpack( X ):
    '''
    Given:
        X: The vector of degrees-of-freedom used by the optimization function representing a 3D line segment.
    Returns:
        startpoint: A length-3 numpy.ndarray containing the ( x,y,z ) of the start point.
        endpoint: A length-3 numpy.ndarray containing the ( x,y,z ) of the end point.
    '''
    return np.asarray( X[:3] ), np.asarray( X[3:] )

def build_energy_for_line_segment( startpoint, endpoint, thresholds, scaffold ):
    '''
    Given:
        startpoint: A sequence containing the ( x,y,z ) of the start point of a line segment.
        endpoint: A sequence containing the ( x,y,z ) of the end point of a line segment.
        thresholds: A dictionary mapping relationships to thresholds.
        scaffold: A data structure containing all the scaffold information. 
    Returns:
        energies: A sequence of functions which take parameters ( startpoint, endpoint )
                  and return a non-negative value measuring how far each constraint
                  is from being met.
    '''
    
    ### 1 Check for endpoints snapping to special points.
    ### 2 Check for parallelism, perpendicularity (start with vertical direction).
    ### 3 Check for point line intersection .
    ### 4 Check for similar lengths.
  
    result = []
    
    ### 1
    for snap_point in scaffold['line_points']:
        ## Python closures are "late binding", so we have to create a
        ## wrapper function to capture each value of `snap_point`.
        ## For more, see: https://docs.python-guide.org/writing/gotchas/

        ## The energies below are similar, we use the wrapper function to 
        ## capture direction, line, and line length.

        ## Without the wrapper function, only the last snap point will be 
        ## remembered.
        ## For example, if we have 2 line_length len1 and len2 need to 
        ## considered, without the wrapper function, only len2 will be 
        ## remembered and this is not correct.
        def consider( snap_point ):
            def point_energy( startpoint, endpoint ):
                return mt.point_point_distance2( snap_point, startpoint )/thresholds['point_point_distance2']
            if point_energy( startpoint, endpoint ) < 1.0:
                result.append( ('point_energy', point_energy) )
        consider( snap_point )
       
        def consider( snap_point ):
            def point_energy( startpoint, endpoint ):
                return mt.point_point_distance2( snap_point, endpoint )/thresholds['point_point_distance2']
            if point_energy( startpoint, endpoint ) < 1.0:
                result.append( ('point_energy', point_energy) )
        consider( snap_point )
    
    ### 2
    for existing_dir in scaffold['line_directions']:
        def consider( existing_dir ):
            def line_parallel_energy( startpoint, endpoint ):
                return mt.line_vec_parallel2( (startpoint, endpoint), existing_dir )/thresholds['line_vec_parallel2']
            if line_parallel_energy( startpoint, endpoint ) < 1.0:
                result.append( ('line_parallel_energy', line_parallel_energy) )
        consider( existing_dir )
        
        def consider( existing_dir ):
            def line_perpendicular_energy( startpoint, endpoint ):
                return mt.line_vec_perpendicular2( (startpoint, endpoint), existing_dir )/thresholds['line_vec_perpendicular2']
            if line_perpendicular_energy( startpoint, endpoint ) < 1.0:
                result.append( ('line_perpendicular_energy', line_perpendicular_energy) )
        consider( existing_dir )
    
    ### 3
    for existing_line in scaffold['construction_lines']:
        def consider( existing_line ):
            def intersection_energy( startpoint, endpoint ):
                return mt.point_line_distance2( startpoint, existing_line )/thresholds['point_line_distance2']
            if intersection_energy( startpoint, endpoint ) < 1.0 :
                result.append( ('intersection_energy', intersection_energy))
        consider( existing_line )

        def consider( existing_line ):
            def intersection_energy( startpoint, endpoint ):
                return mt.point_line_distance2( endpoint, existing_line )/thresholds['point_line_distance2']
            if intersection_energy( startpoint, endpoint ) < 1.0 :
                result.append( ('intersection_energy', intersection_energy))
        consider( existing_line )

    ### 4
    for line_length in scaffold['line_lengths']:
        def consider( line_length ): 
            def length_energy( startpoint, endpoint ):
                return mt.line_length_ratio2( (startpoint, endpoint), line_length )/thresholds['line_length_ratio2']
            if length_energy( startpoint, endpoint ) < 1.0:
                result.append( ('length_energy', length_energy) )
        consider( line_length )
    
    return result

## Iteratively reweighted least squares
def IRLS( startpoint, endpoint, energies ):
    '''
    Given:
        startpoint: A sequence containing the ( x,y,z ) of the start point of a line segment.
        endpoint: A sequence containing the ( x,y,z ) of the end point of a line segment.
        energies: A sequence of functions which take parameters ( startpoint, endpoint )
                  and return a non-negative value measuring how far each constraint
                  is from being met.
    Returns:
        startpoint, endpoint: beautified points
    '''

    # optimize_line_segment function will try to satisfy all the energies remembered from
    # build_energy_for_line_segment. However, it may do optimization in such a way that
    # some energies are partial satisfied, which means the constraint is partiall picked, 
    # this is not correct.
    # 
    # Because the constraint need to be either satisfied, or not satisfied. 
    # that's why we use IRLS to pick the constraints it will satisify.
    

    epsilon = 1e-6
    
    weights = np.ones(len(energies))
    x_previous_iteration = pack( startpoint, endpoint )

    first_iteration = True
    while True:
        startpoint, endpoint = unpack(x_previous_iteration)
        
        for i in range(len(energies)):
            energy_name, energy_func = energies[i]
            value = energy_func(startpoint, endpoint)

            weights[i] = (100 if energy_name == 'point_energy' and first_iteration else 1)/(epsilon + value)
                   
        weights = weights / weights.max()
        # energies: the energies should not be change since we are doing optimize
        # minimize the condition/energies should not change
        result = optimize_line_segment( startpoint, endpoint, energies, weights )
        
        if np.abs(result.x - x_previous_iteration ).sum() < epsilon:
            startpoint, endpoint = unpack(result.x)
            break
        else:
            x_previous_iteration = result.x
            first_iteration = False
    

    # this should not happen
    if not np.allclose(weights, weights[0]):
        thresh = 0.001
        normweights = (weights - weights.min())/(weights.max() - weights.min())
        # # https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
        # # weights might be 1 number, in that case we'll have normweights as nan
        # # np.seterr(divide='ignore', invalid='ignore')
        
        # print('weights', weights)
        # print('normweights', normweights)
        # print( np.any( np.logical_and( normweights > thresh, normweights < 1-thresh ) ) )
        if np.any( np.logical_and( normweights > thresh, normweights < 1-thresh ) ):
            print("DANGER")

    return startpoint, endpoint

## Optimize
def optimize_line_segment( startpoint, endpoint, energies, weights = None ):
    '''
    Given:
        startpoint: A sequence containing the ( x,y,z ) of the start point of a line segment.
        endpoint: A sequence containing the ( x,y,z ) of the end point of a line segment.
        energies: A sequence of functions as would be returned by
                  `build_energy_for_line_segment()`.
        weights (optional): A sequence of floating-point values the same length
                            as `energies` containing the weight for each
                            corresponding function.
    Returns:
        startpoint: The optimized position of the start point.
        endpoint: The optimized position of the end point.
    '''
    
    import scipy.optimize
    
    X0 = pack( startpoint, endpoint )

    if weights is None: weights = np.ones(len(energies))

        
    def E_total( X, debug = False ):
        startpoint, endpoint = unpack( X )
        if debug:
            print('start=', startpoint, '  end=', endpoint)
        total = 0.
        for w, (energy_name, energy_func) in zip( weights, energies ):
            ## Skip energies with 0 weight.
            if w > 0:
                t = w*energy_func( startpoint, endpoint )
                total += t
                if debug:
                    print('term=', t)
        if debug:
            print('total=', total)
            print()
        return total
    E_total(X0, False)
    def callback(xk):
        E_total(xk, False)

    
    # minimize options
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # func - E_total the function we want to minimize
    # X0 - ndarray, shape (n,) initial guess, all ones for our case 
    # method - BFGS, typically requires fewer function calls than the simplex algorithm
    # jac - jacobian, calculate using autograd
    # callback - used to show debug information

    # BFGS options
    # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs
    # disp: Set to True to print convergence messages
    # gtol: Gradient norm must be less than gtol before successful termination.
    # eps: If jac is None the absolute step size used for numerical approximation of the jacobian via forward differences.
    # maxiter: Maximum number of iterations to perform.

    result = scipy.optimize.minimize( E_total,
                                      X0,  
                                      method = 'BFGS', 
                                      jac = grad(E_total), 
                                      tol = 0.000001, 
                                      callback = callback, 
                                      options = { 'disp': False, 'gtol': 0.000001, 'maxiter': 1000 } 
                                      )
    
    # print(result)

    return result


def snap_line_to_other_lines( line, state, thresholds ):
    """
    Given: 
        line: a tuple of 2 points - ((x0, y0, z0), (x1, y1, z1))  
    Returns:
        line: beautified line as np.array
    """
    startpoint, endpoint = line 
    energies = build_energy_for_line_segment( startpoint, endpoint, thresholds, state )

    # print energy_name for debug reasons
    # for (energy_name, energy_func)  in energies:
    #     print(energy_name)
    # print()
    
    # this deals with initially people draw skew line
    try:
        newstart, newend = IRLS( startpoint, endpoint, energies )
    except:
        newstart, newend = startpoint, endpoint

    return np.asarray( [ newstart, newend ] ) 

def add_new_line_info( line, state ):
    """
    Given:
        line: np.array( [start, end] ) 
        state: current scaffold information
    Return:
        no return, add line, line direction, endpoint, midpoint, length to state
    """
    ### 0 Add the line to state['construction_lines'] if not exist.
    ### 1 Add the endpoints, mid-point to state['line_points'], 
    ###   and points assosiated directions to state['points_info'].  
    ### 2 Add line length.
    ### 3 Add line direction.
    p0, p1 = line[0], line[1]
    
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)

    pm = (p0 + p1)/2 # mid point
    line_dir = mt.dir( p1 - p0 ) # line direction
    line_length = mt.point_point_distance( p1, p0 ) # line length

    ### 0
    # empty, add this line anyway
    if len(state['construction_lines']) == 0:
        state['construction_lines'].append( line )

    line_exists = False
    for current_line in state['construction_lines']:
        c0, c1 = current_line[0], current_line[1]
        if ( np.allclose(p0, c0, atol = 1e-4) and np.allclose(p1, c1, atol = 1e-4) ) or \
           ( np.allclose(p0, c1, atol = 1e-4) and np.allclose(p1, c0, atol = 1e-4) ):
            # line exist in state['construction_lines']
            line_exists = True

    if not line_exists:
        state['construction_lines'].append( line )       

    ### 1
    mt.add_point_and_assosiated_directions(p0, line_dir, state)
    mt.add_point_and_assosiated_directions(p1, line_dir, state)
    mt.add_point_and_assosiated_directions(pm, line_dir, state)

    ### 2
    # atol = 1e-3 between line_length, maybe this also could be larger?
    if not any(np.allclose(line_length, x, atol = 1e-3) for x in state['line_lengths']):
        state['line_lengths'].append( line_length )

    ### 3
    # atol = 1.5e-2 for direction, seems quite large
    # use this because atol = 1 - np.cos(10/180*np.pi), parallel energy
    if not any(np.allclose(line_dir,x, atol=0.015) for x in state['line_directions']) and \
       not any(np.allclose(line_dir, -x, atol=0.015) for x in state['line_directions']):
        state['line_directions'].append( line_dir )


