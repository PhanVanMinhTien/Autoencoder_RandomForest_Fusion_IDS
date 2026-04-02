import time
from mininet.net import Mininet

# Connect to running Mininet

from mininet.cli import CLI
from mininet.node import RemoteController, OVSSwitch

def generate_traffic(net):
    h_hr_0 = net.get('h_hr_0')
    h_sale_0 = net.get('h_sale_0')
    h_it_0 = net.get('h_it_0')
    srv = net.get('h_srv_0')

    print("*** Starting traffic generation...")
    srv.cmd('iperf3 -s &')
    srv.cmd('python3 -m http.server 80 &')

    time.sleep(2) # Đợi server khởi động

    while True:
        print("[*] Generating traffic...")

        h_hr_0.cmd('iperf3 -c 192.168.30.10 -t 10 &')
        h_it_0.cmd('iperf3 -c 192.168.30.10 -t 10 &')
        h_sale_0.cmd('curl http://192.168.30.10 &')

        time.sleep(15)

if __name__ == '__main__':
    print("⚠️ Chạy file này trong Mininet CLI bằng py command")