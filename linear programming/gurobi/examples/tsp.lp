\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  102.4499877989256 e[1,0] + 58.9406481131655 e[2,0]
   + 65.37583651472461 e[2,1] + 52.3545604508337 e[3,0]
   + 95.2732911156112 e[3,1] + 87.66413177577246 e[3,2]
   + 41.7612260356422 e[4,0] + 65.60487786742691 e[4,1]
   + 51.478150704935 e[4,2] + 37.48332962798263 e[4,3]
Subject To
 R0: e[1,0] + e[2,0] + e[3,0] + e[4,0] = 2
 R1: e[1,0] + e[2,1] + e[3,1] + e[4,1] = 2
 R2: e[2,0] + e[2,1] + e[3,2] + e[4,2] = 2
 R3: e[3,0] + e[3,1] + e[3,2] + e[4,3] = 2
 R4: e[4,0] + e[4,1] + e[4,2] + e[4,3] = 2
Bounds
Binaries
 e[1,0] e[2,0] e[2,1] e[3,0] e[3,1] e[3,2] e[4,0] e[4,1] e[4,2] e[4,3]
End
