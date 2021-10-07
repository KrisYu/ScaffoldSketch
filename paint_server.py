#!/usr/bin/env python3

## This example comes from: https://websockets.readthedocs.io/en/stable/

import asyncio
from numpy.lib.npyio import load
import websockets
import json
import ssl
import pathlib
import os
from datetime import datetime
from time import strftime

import scaffold_sketch as real_scaffold_sketch
import dummy_scaffold_sketch
scaffold_sketch = real_scaffold_sketch

from copy import deepcopy

def export( state_sequence, gltf_data = None, basename = None, extra_info = None ):
    '''
    Given:
        state_sequence: A sequence of states as in `undo_buffer`.
        gltf_data (optional): If specified, the current state in glTF format.
        basename (optional): If specified, the filename to save into, not including
                             a directory or file extension. If not specified, the current\
                             date and time are used.
        extra_info (optional)
    
    Saves the given sequence of states to the file `~/Desktop/basename.json`
    and `.gltf` if optional glTF data is provided.
    '''
    
    # https://stackoverflow.com/questions/34275782/how-to-get-desktop-location
    desktop = pathlib.Path.home() / 'Desktop'
    # create a folder to store output
    output_dir = desktop / 'VR_output'  
    
    if not output_dir.exists():
        output_dir.mkdir()

    # SimpleDateFormat formatter = new SimpleDateForm
    if basename is None:
        now = datetime.today()
        basename = now.strftime("%Y_%m_%d_%H_%M_%S") + ( "_%d_%s" % ( now.microsecond / 1000, strftime( "%Z" ) ) )


    # save json file
    json_file = os.path.join( output_dir, basename + ".json")
    # https://stackoverflow.com/questions/17043860/how-to-dump-a-dict-to-a-json-file 
    with open(json_file, "w") as f:
        data = [ scaffold_sketch.prepare_state_for_save_all_info(state) for state in state_sequence ]
        if extra_info is not None:
            data.append('start_time' +  extra_info['start_time'])
            print( extra_info['undo_count'] )
            data.append('undo_count' + str(extra_info['undo_count']))
        
        # print(data)
        json.dump( data, f )

    print( "Saved:", json_file )
    
    # GLTF
    if gltf_data is not None:
        gltf_file = os.path.join( output_dir, basename + '.gltf')
        gltf = open(gltf_file, 'wt') 
        gltf.write( gltf_data )
        gltf.close()
        print( "Saved:", gltf_file )

def run( dummy = False, load_state = None ):
    global scaffold_sketch
    if dummy:
        print( "==> Control condition (no-optimization)" )
        scaffold_sketch = dummy_scaffold_sketch
    else:
        scaffold_sketch = real_scaffold_sketch
    
    async def paint_server( websocket, path ):
        # record time at the begining?
        # print( "load_state:", load_state )        
        if load_state is None:
            state = scaffold_sketch.make_new_program_state()
        else:

            try:
                file = open( load_state, 'r' )
                data = json.load( file )
                json_state = data[-3]
                state = scaffold_sketch.load_state_from_json( json_state )
            except:
                print( 'load_state fails ')
                state = scaffold_sketch.make_new_program_state() 

        
        undo_buffer = [ deepcopy( state ) ]
        redo_buffer = []
        
        now = datetime.today()
        start_time = now.strftime("%Y_%m_%d_%H_%M_%S") + ( "_%d_%s" % ( now.microsecond / 1000, strftime( "%Z" ) ) )
    
        undo_count = 0
    
        def save_state_for_undo():
            # print( 'state', state )
            undo_buffer.append( deepcopy( state ) )
            del redo_buffer[:]
    
        
        def undo():
            # https://stackoverflow.com/questions/1281184/why-cant-i-set-a-global-variable-in-python
            nonlocal state
            if len(undo_buffer) > 1:
                # pop last saved state
                redo_buffer.append( undo_buffer.pop() )
                state = deepcopy( undo_buffer[-1] )
                    
        def redo():
            nonlocal state
            if len(redo_buffer) >= 1:
                state = deepcopy( redo_buffer[-1] )
                undo_buffer.append( redo_buffer.pop() )
    
        async for message in websocket:
            parsed = message.split( " ", 1 )
            command = parsed[0]
            
            parameters = None if len( parsed ) == 1 else parsed[1]
            
            if command == "construction-stroke":
                input_data = json.loads( parameters )
                result = scaffold_sketch.line_or_point( state, input_data )
    
                save_state_for_undo()
    
                # result might be None 
                # if no line present and we draw a point
                if not (result is None):
                    if len(result) == 2: # line
                        # may also use https://stackoverflow.com/a/49677241/3608824
                        await websocket.send( "new-straight-line " + json.dumps( result.tolist() ) )
                    elif len(result) == 3: # point
                        await websocket.send("new-point " + json.dumps( result.tolist() ))
    
            elif command == "shape-stroke":
                input_data = json.loads(parameters)
                new_shape = scaffold_sketch.incorporate_new_raw_shape_line( state, input_data )
    
                if not (new_shape is None): 
                    save_state_for_undo()
                    await websocket.send("new-shape-line " + json.dumps( new_shape ))
                else:
                    # this is not a real undo
                    # just clean the not beautified stroke
                    current_state = scaffold_sketch.prepare_state_for_UI( state )
                    await websocket.send("undo " +  json.dumps( current_state ) )
            
            # load from existing state
            elif command == "load-state" and load_state is not None:
                current_state = scaffold_sketch.prepare_state_for_UI( state )
                await websocket.send("load-state " + json.dumps( current_state ) )

            elif command == "undo":
                undo()
                undo_count += 1
                current_state = scaffold_sketch.prepare_state_for_UI( state )
                #print('state after undo', state)
                await websocket.send("undo " +  json.dumps( current_state ) )
            
            elif command == "redo":
                redo()
                current_state = scaffold_sketch.prepare_state_for_UI( state )
                #print('state after redo', current_state)
                await websocket.send("redo " + json.dumps( current_state ) )
            


            
            elif command == "export":
                export( undo_buffer, gltf_data = parameters, extra_info = {'start_time': start_time, 'undo_count': undo_count} )
            
            ## Always save, overwriting the files named 'last_session'
            ## If export() is slow, pass `undo_buffer[-1:]`
            export( undo_buffer[-1:], basename = 'last_session' )
    
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert_pem = "../cert.pem"
    ssl_context.load_cert_chain(cert_pem)
    start_server = websockets.serve(
        paint_server, None, 9000, ssl=ssl_context, max_size = None
        )
    
    #start_server = websockets.serve( paint_server, None, 9000 )
    
    asyncio.get_event_loop().run_until_complete( start_server )
    asyncio.get_event_loop().run_forever()
