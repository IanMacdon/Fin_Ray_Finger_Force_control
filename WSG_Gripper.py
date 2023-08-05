import socket
from time import time, sleep
import atexit
import struct
from textwrap import wrap
import matplotlib.pyplot as plt


class Gripper:
    def __init__(self, TCP_IP="192.168.1.20", TCP_PORT=1000):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT
        self.BUFFER_SIZE = 1024
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.timeout = 2
        self.last_pos = 109
        atexit.register(self.__del__)
        # Acknowledge fast stop from failure if any
        #self.ack_fast_stop()
        self.preamble = b'\xAA\xAA\xAA'
        self.check_sum = b'\x33\x35'

    def __del__(self):
        self.tcp_sock.close()

    def send_speed(self, value):
        pram = b'\xAA\xAA\xAA\xBB\x04\x00'  # Includes the preamble and, the command and the length, should split up better
        check = b'\x33\x35'  # Random checksum, does not actually check
        payload = struct.pack('<f', value)  # Convert to float, use little endian
        data = pram + payload + check  # add all together
        self.tcp_sock.send(data)
        #self.tcp_sock.recv(self.BUFFER_SIZE)

    def read_pos(self):
        msg = self.preamble + b'\xBC' + b'\x00\x00' + self.check_sum
        self.tcp_sock.send(msg)
        data = self.tcp_sock.recv(self.BUFFER_SIZE)
        payload_length = data[4]
        packed_data = data[6:6 + payload_length]
        #print(packed_data)
        try:
            val = struct.unpack('f', packed_data)
        except:
            val = [self.last_pos]
        self.last_pos = val[0]
        return val[0]

    def homing_gripper(self):
        msg = self.preamble + b'\xBD' + b'\x00\x00' + self.check_sum
        self.tcp_sock.send(msg)
        sleep(2)

