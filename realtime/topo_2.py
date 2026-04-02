from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

from mininet.node import Node 


def myNetwork():
    net = Mininet(topo=None, build=True, ipBase='192.168.1.0/24')

    # Adding Controller
    print("*** Adding Controller")
    net.addController(name='c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    # Adding Switches
    print("*** Adding Switches")
    core1 = net.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
    core2 = net.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')
    
    # Access Switches: s3 (Sale), s4 (HR), s5 (IT), s6 (Server)
    access_switches = []
    for i in range(3, 7):
        sw = net.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
        access_switches.append(sw)

    # Adding Router (for inter-VLAN routing)
    

    ####


    # Adding gateway router (R1) for inter-VLAN routing
 
    gw = net.addHost('gw')

    # 4 link tương ứng 4 VLAN
    net.addLink(gw, core1, port1=0, port2=10, intfName1='gw-eth0')
    net.addLink(gw, core1, port1=1, port2=11, intfName1='gw-eth1')
    net.addLink(gw, core1, port1=2, port2=12, intfName1='gw-eth2')
    net.addLink(gw, core1, port1=3, port2=13, intfName1='gw-eth3')
    ####


    # Adding Hosts and Links
    print("*** Adding Hosts")
    # VLAN 10 (Sale) -> Switch 3
    sale_hosts = [net.addHost(f'h_sale_{i}', ip=f'192.168.10.{i+10}/24', defaultRoute='via 192.168.10.1') for i in range(3)]
    # VLAN 20 (HR) -> Switch 4
    hr_hosts = [net.addHost(f'h_hr_{i}', ip=f'192.168.20.{i+10}/24', defaultRoute='via 192.168.20.1') for i in range(3)]
    # VLAN 90 (IT) -> Switch 5
    it_hosts = [net.addHost(f'h_it_{i}', ip=f'192.168.90.{i+10}/24', defaultRoute='via 192.168.90.1') for i in range(3)]
    # VLAN 30 (Server Farm) -> Switch 6
    srv_hosts = [net.addHost(f'h_srv_{i}', ip=f'192.168.30.{i+10}/24', defaultRoute='via 192.168.30.1') for i in range(2)]

    print("*** Creating Links (Strict Port Mapping)")
    for i, sw in enumerate(access_switches):
        net.addLink(sw, core1, port1=1, port2=i+1) # eth1 nối Core 1
        net.addLink(sw, core2, port1=2, port2=i+1) # eth2 nối Core 2

    net.addLink(core1, core2, port1=5, port2=5) # Link dự phòng giữa 2 Core

    # Nối Hosts vào Switch Access (bắt đầu từ eth3)
    for i, h in enumerate(sale_hosts): net.addLink(h, access_switches[0], port2=i+3)
    for i, h in enumerate(hr_hosts):   net.addLink(h, access_switches[1], port2=i+3)
    for i, h in enumerate(it_hosts):   net.addLink(h, access_switches[2], port2=i+3)
    for i, h in enumerate(srv_hosts):  net.addLink(h, access_switches[3], port2=i+3)




    print("*** Starting Network")
    net.start()
    # net.get
    gw = net.get('gw')
    h_hr_0 = net.get('h_hr_0')
    h_sale_0 = net.get('h_sale_0')
    h_it_0 = net.get('h_it_0')

    # Gán IP cho từng VLAN
    gw.cmd('ifconfig gw-eth0 192.168.10.1/24')
    gw.cmd('ifconfig gw-eth1 192.168.20.1/24')
    gw.cmd('ifconfig gw-eth2 192.168.90.1/24')
    gw.cmd('ifconfig gw-eth3 192.168.30.1/24')

    # Enable routing
    gw.cmd('sysctl -w net.ipv4.ip_forward=1')


    print("*** Generating traffic automatically...")
    import time

    srv = net.get('h_srv_0')

    srv.cmd('iperf3 -s &')
    srv.cmd('python3 -m http.server 80 &')
    time.sleep(2)  # Đợi server khởi động
    h_hr_0.cmd('iperf3 -c 192.168.30.10 -t 20 &')
    h_sale_0.cmd('curl http://192.168.30.10 &')
    h_it_0.cmd('iperf3 -c 192.168.30.10 -t 20 &')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    myNetwork()