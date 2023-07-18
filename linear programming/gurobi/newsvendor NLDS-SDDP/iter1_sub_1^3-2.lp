\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^3 + 10 B_1^3 + theta_3^3 - 120 Constant
Subject To
 R0: theta_3^3 >= 0
 R1: I_1^3 - B_1^3 = -2
 R2: 10 q_2^3 + 10 I_1^3 - 10 B_1^3 + theta_3^3 >= 100
 R3: 10 q_2^3 + 10 I_1^3 - 10 B_1^3 + theta_3^3 >= 100
 R4: 10 q_2^3 + 10 I_1^3 - 10 B_1^3 + theta_3^3 >= 100
 R5: 10 q_2^3 + 10 I_1^3 - 10 B_1^3 + theta_3^3 >= 100
Bounds
 Constant = 1
End
