\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^2 + 10 B_1^2 + theta_3^2 - 90 Constant
Subject To
 R0: theta_3^2 >= 0
 R1: I_1^2 - B_1^2 = 1
 R2: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R3: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R4: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R5: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
Bounds
 Constant = 1
End
