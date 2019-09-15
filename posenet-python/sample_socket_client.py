from websocket import *
ws = create_connection("ws://66.31.16.203:5000")
print("Sending 'Hello, World'...")
ws.send("Hello, World")
result =  ws.recv()
ws.close()