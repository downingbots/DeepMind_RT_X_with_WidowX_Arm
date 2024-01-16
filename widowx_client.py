#!/usr/bin/env python
import os
import array
import time
import struct
from threading import Thread
import serial
import os
import json
import regex as re
import shutil
import subprocess
import numpy as np

from threading import Timer
from ctypes import c_uint8 as unsigned_byte

class WidowX(object):

    def __init__(self):
        self.config = self.read_config()
        success = False
        self.running = True
        self.debug = False
        self.state = {}
        time.sleep(1)
        while not success:
          try:
            self.widowx = serial.Serial('/dev/ttyUSB0', 115200, timeout=40)
          except:
            self.widowx = serial.Serial('/dev/ttyUSB1', 115200, timeout=40)
          [success, err_msg] = self.wait_for_ok(True)
        print("read ok")
        time.sleep(1)
        self.widowx.write(b'ok\n')
        time.sleep(1)

        # print("Final handshake.")
        # self.wait_for_ok(True)
        # print("Press PS Button to start!")

    def read_config(self):
        with open('rt1_widowx_config.json') as config_file:
          config_json = config_file.read()
        config = json.loads(config_json)
        return config

    def reset_usb(self):
        reset_names = [self.config["arm_usb"]]

        if shutil.which('usbreset') is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            res = subprocess.call(f'gcc {current_dir}/usbreset.c -o /usr/local/bin/usbreset')
            if not res == 0:
                print(f'usbreset install exit code: {res}')
                raise ValueError('could not install usbreset !')
        res = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
        lines = res.stdout.decode().split('\n')
        for line in lines:
            print("lines:", line, reset_names)
            for name in reset_names:
                if name in line:
                    numbers = re.findall(r'(\d\d\d)', line)[:2]
                    print('resetting usb with lsusb line: {}'.format(line))
                    cmd = 'sudo usbreset /dev/bus/usb/{}/{}'.format(*numbers)
                    print(cmd)
                    res = subprocess.call(cmd.split())
                    if not res == 0:
                        print(f'exit code: {res}')
                        raise ValueError('could not reset !')
                    print("Usb reset")


    def wait_for_ok(self, wait_for_state = False):
        print("wait for ok", wait_for_state)
        success = True 
        err_msg = None
        got_ok = got_err = got_state = False
        while True:
          line = self.widowx.read_until('\n'.encode())
          if len(line) == 0 or line[-1:] != '\n'.encode():
            print("Timeout line:", line)
            print("", line[-1:], '\n'.encode())
            if line[-1:] == '\n'.encode():
              print("same")
            success = False
            if got_ok:
              err_msg = "Timeout waiting for state" 
            elif got_state:
              err_msg = "Timeout waiting for ok" 
            else:
              err_msg = "Timeout" 
            print(err_msg)
            exit()
            # reset_usb didn't end up resolving any disconnect issues.
            # self.reset_usb()
            # return [success, err_msg]
          elif(line == 'ok\n'.encode() or line == 'ok\r\n'.encode()):
            got_ok = True
          elif(line == 'No solution for IK!\n'.encode()):
            got_err = True
            err_msg = line
          elif(line[:6] == 'State:'.encode()):
            got_state = True
            # line is a string of unsigned bytes (char) and commas
            for i,char in enumerate(line[6:].split(b',')):
              # note: self.state is not normalized.
              if i == 0:
                self.state["x"] = int(char) & 0x7F
                if int(char) >> 7: 
                  self.state["x"] -= 128
              elif i == 1:
                self.state["y"] = int(char) & 0x7F
                if int(char) >> 7: 
                  self.state["y"] -= 128
              elif i == 2:
                self.state["z"] = int(char) & 0x7F
                if int(char) >> 7: 
                  self.state["z"] -= 128
              elif i == 3:
                self.state["gamma"] = int(char)
              elif i == 4:
                self.state["rot"] = int(char)
              elif i == 5:
                lB = int(char)
            if ((lB >> 7) & 1):
              self.state["gamma"] = -self.state["gamma"] 
            if ((lB >> 6) & 1):
              self.state["rot"] = -self.state["rot"] 
            if ((lB >> 5) & 1):
              self.state["gripper"] = 1
            elif ((lB >> 4) & 1):
              self.state["gripper"] = 0
            else:
              self.state["gripper"] = None
              print("no gripper state", lB)
            print("got state: ", self.state)
          else:
            print("line: ", line)
            # get next line
          if got_ok and not wait_for_state:
            print("got ok")
            success = True
            break
          if got_ok and got_err:
            print("got err", err_msg)
            success = False
            break
          elif got_ok and got_state:
            success = True
            break
        print("ready")
        return [success, err_msg]

    def send_msg(self, vx,vy,vz, vgamma, vq5, lB):
        msg = "%d,%d,%d,%d,%d,%d" %(vx,vy,vz, vgamma, vq5, lB)
        print("msg",msg)
        while True:
          if not self.debug and self.widowx.in_waiting:
            xs = [np.uint8(vx),np.uint8(vy),np.uint8(vz), np.uint8(vgamma), np.uint8(vq5), np.uint8(lB)]
            self.widowx.write(xs)
            break
          else:
            print("msg not sent")
            time.sleep(1)

    def move_to_position(self, pos_name):
        if pos_name == "Rest":
            print("moveRest")
            option = 0x01    # moveRest
        elif pos_name == "Home":
            print("moveHome")
            option = 0x02    # moveHome
        elif pos_name == "Center": 
            # for bw compat, make Center == Home
            # 0x03 now uses by To Point
            print("moveHome")
            option = 0x02    # moveHome
        elif pos_name == "To Point":
            print("move to point")
            option = 0x03    # move center
        elif pos_name == "By Point":
            print("move by point")
            option = 0x06    # Move by point
        elif pos_name == "Fixed Wrist Angle":
            print("move with fixed wrist angle")
            option = 0x07    # Move arm from {1}
        elif pos_name == 'Relax':
            print("relaxServos")
            option = 0x04
        elif pos_name == 'Torque':
            print("torqueServos")
            option = 0x05
        else:
            print("unknown position")
            option = 0x00
        self.send_msg(0, 0, 0, 0, 0, unsigned_byte(option).value)
        # if option in [1,2,4,5]:
        #   [success,err_msg] = self.wait_for_ok(True)
        # else:
        #   [success,err_msg] = self.wait_for_ok(False)
        [success,err_msg] = self.wait_for_ok(True)
        return [success, err_msg]
 

    # [-255,255] for vx, vy 
    # [-127,127] for vz and [-255,255] for vg, vg_rot
    # -255 < wrist_rotate velocity < 255
    # -255 < wrist_angle_velocity < 255
    # gripper_open = {True, False}
    def move(self, vx, vy, vz, vg_angle, vg_rot, gripper_open):
        #  Message byte index --> data
        #  0 --> speed in x, where MSb is sign
        #  1 --> speed in y, where MSb is sign
        #  2 --> speed in z, where MSb is sign
        #  3 --> speed for gamma
        #  4 --> speed for q5
        #  5 --> Sg<<7 | Sq5 <<6 | open_close[1..0] << 4 | options[3..0]
        #          Sg --> sign of gamma speed
        #          Sq5 --> sign of Q5 speed
        #          open_close: 0b00 || 0b11 --> void
        #                      0b01 --> open
        #                      0b10 --> close
        #          options: 1 --> rest
        #                   2 --> home
        #                   3 --> center
        #                   4 --> relax
        #                   5 --> torque
        #                   6 --> moveOption = POINT_MOVEMENT
        #                   7 --> moveOption = USER_FRIENDLY
        #                    other --> void

        # abs(vg_angle) and abs(vq) values <= 255 using up 8 bits.
        # signs are sent as separate bits in lB.
        lB = 0x00
        if vg_angle < 0:
          lB = (0x01 << 7)
        else:
          lB = 0x00
        vg = abs(vg_angle)
        if vg_rot < 0:
          lB |= (0x01 << 6)
        #   vq5 = abs(vg_rot)
        # else:
        #   vq5 = -vg_rot
        # vq5 = vg_rot
        vq5 = abs(vg_rot)

        if gripper_open:
          lB |= (0x01 << 4)
        elif not gripper_open:
          lB |= (0x01 << 5)
        # elif gripper_open is None:
        #   lB = 0x00
        print("lB:", unsigned_byte(lB).value)

        # self.send_msg(vx,vy,vz, vg_angle, vq5, unsigned_byte(lB).value)
        self.send_msg(vx,vy,vz, vg, vq5, unsigned_byte(lB).value)
        # if (vx or vy or vz or vg):
        #   print("prewait:", vx, vy, vz, vg)
        #   [success,err_msg] = self.wait_for_ok(True)
        # else:
        #   [success,err_msg] = [True, None]
        [success,err_msg] = self.wait_for_ok(True)
        return [success, err_msg]
