# from mininet.net import Mininet
# from mininet.node import RemoteController, OVSSwitch
# from mininet.cli import CLI
# from mininet.log import setLogLevel

# def myNetwork():

#     #########################
#     #       IP base         #
#     #########################
#     net = Mininet(topo=None, build=True, ipBase='192.168.1.0/24')

#     ##########################
#     # Adding Controller Code #
#     ##########################
#     print("*** Adding Controller")
#     net.addController(name='c0', controller=RemoteController, ip='127.0.0.1', port=6653)


#     ##########################
#     # Adding Hosts Code      #
#     ##########################
#     print("*** Adding Hosts")

#     # SALE VLAN 10
#     sale_hosts = []
#     for i in range(0, 3): 
#         # Cần chữ 'f' trước chuỗi IP để Python hiểu biến {i}
#         host_name = f'h_sale_{i}'
#         host_ip = f'192.168.10.{i+10}/24'
        
#         sale_host = net.addHost(host_name, ip=host_ip, defaultRoute='via 192.168.10.1')

#         # PHẢI CÓ DÒNG NÀY để đưa host vào group
#         sale_hosts.append(sale_host)

#         print(f"*** Created {host_name} with IP {host_ip}")
        
#     #h1 = net.addHost('h1', ip='192.168.10.10/24')
#     #h2 = net.addHost('h2', ip='192.168.10.11/24')
#     #h3 = net.addHost('h3', ip='192.168.10.12/24')


#     # HR VLAN 20
#     hr_hosts = []
#     for i in range(0, 3): 
#         host_name = f'h_hr_{i}'
#         host_ip = f'192.168.20.{i+10}/24'
        
#         hr_host = net.addHost(host_name, ip=host_ip, defaultRoute='via 192.168.20.1')
        
#         hr_hosts.append(hr_host)

#         print(f"*** Created {host_name} with IP {host_ip}")
        
#     #h4 = net.addHost('h4', ip='192.168.20.10/24')
#     #h5 = net.addHost('h5', ip='192.168.20.11/24')
#     #h6 = net.addHost('h6', ip='192.168.20.12/24')

#     # IT VLAN 90

#     it_hosts = []
#     for i in range (0,3):
#         host_name = f'h_it_{i}'
#         host_ip = f'192.168.90.{i+10}/24'

#         it_host = net.addHost(host_name, ip=host_ip, defaultRoute='via 192.168.90.1')

#         it_hosts.append(it_host)

#         print(f"*** Created {host_name} with IP {host_ip}")

    

#     ##########################
#     # Adding Switches Code   #
#     ##########################
#     print("*** Adding Switch")
#     core1 = net.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
    
#     core2 = net.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')

#     # Cấu hình cho lớp Access bằng vòng lặp
#     access_switches = []
#     for i in range(3, 7):
#         sw = net.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
#         access_switches.append(sw)

#     ##########################
#     #   Adding Links Code    #
#     ##########################
#     print("*** Creating Links with strict Port Mapping")

#     # Quy hoạch: 
#     # Access (s3-s6) dùng eth1 nối Core1 (s1), eth2 nối Core2 (s2)
#     # Core (s1/s2) dùng eth1->eth4 để nhận kết nối từ s3->s6 tương ứng.

#     for i, sw in enumerate(access_switches):
#         # i=0 (s3), i=1 (s4), i=2 (s5), i=3 (s6)
        
#         # Kết nối tới Core 1 (s1)
#         # sw-eth1 nối với s1-eth(i+1)
#         net.addLink(sw, core1, port1=1, port2=i+1)
        
#         # Kết nối tới Core 2 (s2)
#         # sw-eth2 nối với s2-eth(i+1)
#         net.addLink(sw, core2, port1=2, port2=i+1)

#     # Nối 2 Core Switch với nhau (Dùng cổng eth5 cho chuyên nghiệp)
#     net.addLink(core1, core2, port1=5, port2=5)

#     # Nối Hosts vào các cổng còn lại (bắt đầu từ eth3)
#     # Sale VLAN 10 nối vào s3 (access_switches[0])
#     for i, host in enumerate(sale_hosts):
#         net.addLink(host, access_switches[0], port2=i+3)

#     # HR VLAN 20 nối vào s4 (access_switches[1])
#     for i, host in enumerate(hr_hosts):
#         net.addLink(host, access_switches[1], port2=i+3)

#     # IT VLAN 90 connect to s5 (access_switches[2])

#     for i, host in enumerate(it_hosts):
#         net.addLink(host, access_switches[2], port2=i+3)


#     print("*** Starting Network")
#     #net.build()
#     net.start()
    
#     print("\n[*] Topology doanh nghiệp đã sẵn sàng!")
#     print("[*] VLAN 10: 192.168.10.10-12 | VLAN 20: 192.168.20.10-12 | VLAN 90: 192.168.30.10-12")
    
#     # Kích hoạt chế độ mirroring để trích xuất đặc trưng (Tùy chọn nâng cao)
#     # Hoặc đơn giản là lắng nghe trên interface của Switch
    
#     CLI(net)
#     net.stop()

# if __name__ == '__main__':
#     setLogLevel('info')
#     myNetwork()

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

def myNetwork():
    net = Mininet(topo=None, build=True, ipBase='192.168.1.0/24')

    print("*** Adding Controller")
    net.addController(name='c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    print("*** Adding Switches")
    core1 = net.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
    core2 = net.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')
    
    # Access Switches: s3 (Sale), s4 (HR), s5 (IT), s6 (Server)
    access_switches = []
    for i in range(3, 7):
        sw = net.addSwitch(f's{i}', cls=OVSSwitch, protocols='OpenFlow13')
        access_switches.append(sw)

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
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    myNetwork()