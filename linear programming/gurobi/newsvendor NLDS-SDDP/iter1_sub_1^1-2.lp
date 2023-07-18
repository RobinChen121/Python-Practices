\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^1 + 10 B_1^1 + theta_3^1 - 50 Constant
Subject To
 R0: theta_3^1 >= 0
 R1: I_1^1 - B_1^1 = 5
 R2: 10 q_2^1 + 10 I_1^1 - 10 B_1^1 + theta_3^1 >= 100
 R3: 10 q_2^1 + 10 I_1^1 - 10 B_1^1 + theta_3^1 >= 100
 R4: 10 q_2^1 + 10 I_1^1 - 10 B_1^1 + theta_3^1 >= 100
 R5: 10 q_2^1 + 10 I_1^1 - 10 B_1^1 + theta_3^1 >= 100
Bounds
 Constant = 1
End
