import os
import subprocess
import time
import json


class LeapDecoder(object):
    def __init__(self):
        self.r_send, self.w_send = os.pipe()
        self.r_recv, self.w_recv = os.pipe()
        os.set_inheritable(self.r_send, True)
        os.set_inheritable(self.w_recv, True)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.daemon = subprocess.Popen(['leapd'])
        self.encoder = subprocess.Popen([
                'python2', os.path.join(dir_path, 'leap_encoder.py'),
                f'{self.r_send}', f'{self.w_recv}'
            ], close_fds=False)
        print('waiting for deamon to start...')
        delay = 10
        for i in range(delay):
            print(f"Sleeping for {delay - i} seconds...")
            time.sleep(1)

    def __del__(self):
        self.send('close')
        os.close(self.w_send)
        os.close(self.r_recv)
        self.daemon.terminate()

    def send(self, msg):
        if self.encoder.poll() is not None:
            raise ValueError("LeapDecoder as terminated.")
        length = f'{len(msg):05}'
        os.write(self.w_send, length.encode('utf-8'))
        os.write(self.w_send, msg.encode('utf-8'))

    def receive(self):
        length = os.read(self.r_recv, 5).decode('utf-8')
        msg = os.read(self.r_recv, int(length)).decode('utf-8')
        return json.loads(msg)


class LeapCamera(object):
    def __init__(self):
        self.comms = LeapDecoder()

    def render(self):
        self.comms.send('render')

    def detect(self):
        self.comms.send('detect')
        return self.comms.receive()


if __name__ == '__main__':
    leap = LeapCamera()
    leap.render()
    for _ in range(10):
        t = time.time()
        print(leap.detect())
        print(time.time() - t)
        time.sleep(1)
