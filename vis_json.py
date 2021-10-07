import json
import sys

import numpy as np
import polyscope as ps


def read_data_from_file( filename ):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def vis_lines( lines, name = 'lines' ):
    '''
    https://polyscope.run
    Given:
        lines format
        [
            [ [ x0, y0, z0], [ x1, y1, z1] ], # p0, p1 for line1
            [ [ x2, y2, z2], [ x3, y3, z3] ], # p2, p3 for line2
            ...
        ]
    Return:
        no return
    '''

    # nodes all points, format
    #  array([
    #     [ x0,  y0, z0 ], #p0 for line1
    #     [ x1,  y1, z1 ], #p1 for line1
    #     [ x2,  y2, z2 ], #p2 for line2
    #     [ x3,  y3, z3 ], #p3 for line2
    #     ...
    #   ])
    nodes = np.asarray( lines ).reshape(-1, 3)
    
    # edges, format
    # array([
    #         [ 0,  1],
    #         [ 2,  3],
    #         ...
    #      ])
    edges = np.arange( len(nodes) ).reshape(-1, 2)
    
    # visualize
    ps_net = ps.register_curve_network( name , nodes, edges)


def vis_curves( curves, name = 'curves' ):
    '''
    curves format
        [
            [ [ x0, y0, z0], ...], # pts for curve1
            [ [ x1, y1, z1], ...], # pts for curve2
            ...
        ]
    Return:
        no return
    '''
    nodes = np.zeros((0, 3))
    edges = np.zeros((0, 2))


    pt_cnt = 0

    for curve in curves:
        curve_points = np.asarray( curve ).reshape(-1, 3)

        curve_connects = np.zeros( (len(curve_points)-1, 2) )
        curve_connects[:,0] = np.arange( pt_cnt     , pt_cnt + len(curve_points) - 1)
        curve_connects[:,1] = np.arange( pt_cnt + 1 , pt_cnt + len(curve_points) )

        pt_cnt += len(curve_points)


        nodes = np.concatenate((nodes, curve_points) )
        edges = np.concatenate((edges, curve_connects))
    
    ps_net = ps.register_curve_network(name , nodes, edges )





# main 
ps.init()
filename = sys.argv[1]
if len(sys.argv) < 2:
    print('usage: read_output_json.py output_json_filename.json [raw_lines|lines|raw_curves|curves]')
    exit()

data = read_data_from_file( filename )
print(data[-1])
print(data[-2])

try:
    raw_lines = data[-3]['raw_construction_lines']
    lines = data[-3]['construction_lines']

    # add points
    line_points = data[-3]['line_points']
    raw_curves = data[-3]['raw_shape_curves']
    curves = data[-3]['shape_curves']

    vis_lines( raw_lines, 'raw_construction_lines' )
    vis_lines( lines, 'construction_lines' )
    vis_curves(raw_curves, 'raw_shape_curves')
    vis_curves(curves, 'shape_curves')
    ps.register_point_cloud('points', np.asarray(line_points))

except Exception as e:
    lines = data[-3]['construction_lines']
    curves = data[-3]['shape_curves']
    vis_lines( lines, 'construction_lines' )
    vis_curves(curves, 'shape_curves')
else:
    pass
finally:
    ps.show()






    

