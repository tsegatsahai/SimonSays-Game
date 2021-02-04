import socket

host = '127.0.0.1'
port = 65535
s = socket.socket()

s.connect((host, port))
f = open('10.jpg', 'rb')
print('Sending...')
img = f.read(2048)
while img:
    print('Sending...')
    s.send(img)
    img = f.read(2048)
f.close()
print("Done Sending")
s.shutdown(socket.SHUT_WR)
# print(s.recv(2048))

poseDetected = s.recv(8)

# single player
poseDetected = int(bytes.decode(poseDetected, 'utf-8'))

# multiplayer
# poseDetected = (bytes.decode(poseDetected, 'utf-8')).split(',')

s.close()

