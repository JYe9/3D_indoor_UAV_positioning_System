import cv2
import depthai as dai
from birdseyeview import BirdsEyeView
from camera import Camera
from typing import List
import config
import socket
import threading
import time
from camera import Camera
import socketserver
from collections import deque

HEADER = 64
PORT = 11101
# detectedDistance = 123.0  # distance to be sent to MATLAB

# SERVER = socket.gethostbyname(socket.gethostname())
SERVER = "127.0.0.1"
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"
distance_queue = deque(maxlen=3)

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(
    key=lambda x: x.getMxId(), reverse=True
)  # sort the cameras by their mxId
friendly_id = 1
device_info = device_infos[0]
camera = Camera(device_info, friendly_id, show_video=True)


class RequestHandler(socketserver.StreamRequestHandler):
    # timeout = 10

    def handle(self):
        global distance_queue

        print(f"[NEW CONNECTION] {self.client_address} connected.")
        connected = True
        while connected:
            # print(f"{distance_queue = }")
            msg_length = self.rfile.read(HEADER).decode(FORMAT)
            if msg_length:
                msg_length = int(msg_length)
                msg = self.rfile.read(msg_length).decode(FORMAT)
                if msg == DISCONNECT_MESSAGE:
                    connected = False

                print(f"[{self.client_address}] {msg}")
                self.wfile.write("Msg received".encode(FORMAT))

                try:
                    distance = distance_queue.popleft()
                except IndexError:
                    pass
                # else:  # No error popup
                self.wfile.write(f"{distance}".encode(FORMAT))


            else:
                connected = False

    # def finish(self):
    #     print(f"[DISCONNECTION] {self.client_address}.")
    #     super().finish()


def sendDistance():
    global distance_queue, camera
    while True:
        detectedDistance = camera.update()
        if detectedDistance is not None:
            distance_queue.append(detectedDistance + 1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


def TCPstart():
    with socketserver.TCPServer(ADDR, RequestHandler, bind_and_activate=False) as tcpSerSock:
        tcpSerSock.allow_reuse_address = True
        tcpSerSock.server_bind()
        tcpSerSock.server_activate()
        # tcpSerSock.serve_forever()
        print(f"[LISTENING] Server is listening on {SERVER}")

        threading.Thread(target=tcpSerSock.serve_forever, daemon=True).start()

        sendDistance()


if __name__ == "__main__":
    print("[STARTING] server is starting...")
    TCPstart()
