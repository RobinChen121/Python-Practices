\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^2 + 10 B_1^2 + theta_3^2 - 90 Constant
Subject To
 R0: theta_3^2 >= 0
 R1: I_1^2 - B_1^2 = 4
 R2: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R3: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R4: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R5: 10 q_2^2 + 10 I_1^2 - 10 B_1^2 + theta_3^2 >= 100
 R6: 4 q_2^2 + 4 I_1^2 - 4 B_1^2 + theta_3^2 >= 52
 R7: 4 q_2^2 + 4 I_1^2 - 4 B_1^2 + theta_3^2 >= 52
 R8: 2 q_2^2 + 2 I_1^2 - 2 B_1^2 + theta_3^2 >= 29
 R9: theta_3^2 >= 0
Bounds
 Constant = 1
End
