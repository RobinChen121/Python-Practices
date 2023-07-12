\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_1 - 0.003 W_0 + 0.03 W_1 + theta_2 + 25 Constant
Subject To
 R0: theta_2 >= -1200
 R1: - q_1 + 0.003 W_0 - 0.03 W_1 >= -105
 R2: - q_1 - W_0 + W_1 = -5
Bounds
 theta_2 free
 Constant = 1
End
