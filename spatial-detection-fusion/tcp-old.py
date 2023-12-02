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


HEADER = 64
PORT = 11101
detectedDistance = 145.0  # distance to be sent to MATLAB

# SERVER = socket.gethostbyname(socket.gethostname())
SERVER = "127.0.0.1"
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)


device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(
    key=lambda x: x.getMxId(), reverse=True
)  # sort the cameras by their mxId
friendly_id = 0
device_info = device_infos[0]
camera = Camera(device_info, friendly_id, show_video=True)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False

            print(f"[{addr}] {msg}")
            conn.send("Msg received".encode(FORMAT))

    conn.close()


def sendDistance():
    global distance
    while True:
        detectedDistance = camera.update()
        if detectedDistance is not None:
            distance = detectedDistance + 1


def TCPstart():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        thread2 = threading.Thread(target=sendDistance)
        thread2.start()
        t = conn.sendall(f"{distance}".encode())


if __name__ == "__main__":
    print("[STARTING] server is starting...")
    TCPstart()
