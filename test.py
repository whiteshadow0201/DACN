
import math
def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

def inverse_cantor(z):
    w = int((math.sqrt(8*z + 1) - 1) // 2)
    t = (w * (w + 1)) // 2
    k2 = z - t
    k1 = w - k2
    return k1, k2
k=[]

for i in range (7):
    for j in range (7):
        if ((i==6) and (j<6) or (j==6) and (i<6)):
            k.append(cantor_pairing(i, j))



node = "Node1"
honeypot = f"Honeypot{{{node}}}"
print(honeypot)