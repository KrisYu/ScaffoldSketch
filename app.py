import asyncio
import http.server
import os
import socket    
import socketserver
import ssl
import threading
import tkinter as tk
import sys
import argparse

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def start_http_server(path, port):
	os.chdir(path)
	Handler = http.server.SimpleHTTPRequestHandler
	Handler.extensions_map.update({
		".js": "application/javascript",
	});
	with socketserver.TCPServer(("", port), Handler) as httpd:
		httpd.serve_forever()

def start_https_server(path, port):
    os.chdir(path)
    Handler = http.server.SimpleHTTPRequestHandler
    Handler.extensions_map.update({
        ".js": "application/javascript",
    });
    httpd = http.server.HTTPServer(("", port), Handler)
    httpd.socket = ssl.wrap_socket(httpd.socket, 
        certfile='../cert.pem', server_side=True)
    httpd.serve_forever()

def main():
    arg_parser = argparse.ArgumentParser( description='ScaffoldSketch' )
    arg_parser.add_argument('--dummy', action='store_true', help='Disable auto-correct.')
    arg_parser.add_argument('--load-state', type=str, help='A path to a JSON file with initial state to load.' )
    args = arg_parser.parse_args()
    
    path = "html/"
    port = 4443
    
    ipaddress = get_ip_address()
    url = "https://{}:{}/".format(ipaddress, port)
    print("Serving at {}".format(url))
    
    # Start the web server
    web_server = threading.Thread(name='web_server',
                                   target=start_https_server,
                                   args=(path, port))
    web_server.setDaemon(True)
    web_server.start()
    
    def start_app_server():
        asyncio.set_event_loop(asyncio.new_event_loop())
        import paint_server
        paint_server.run( **vars(args) )
    
    # Start a separate app thread here.
    app_server = threading.Thread(name='app_server',
                                  target=start_app_server)
    app_server.setDaemon(True)
    app_server.start()
    
    # App GUI window gets the main loop
    root = tk.Tk()
    root.title("App Server")
    tk.Label(root, text="Serving at {}".format(url)).pack()
    tk.Button(root, text="Quit", command=root.destroy).pack()
    tk.mainloop()
    
if __name__ == '__main__':
    main()
