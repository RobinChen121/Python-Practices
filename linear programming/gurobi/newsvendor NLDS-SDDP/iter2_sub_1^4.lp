\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^4 + 10 B_1^4 + theta_3^4 - 160 Constant
Subject To
 R0: theta_3^4 >= 0
 R1: I_1^4 - B_1^4 = -3
 R2: 10 q_2^4 + 10 I_1^4 - 10 B_1^4 + theta_3^4 >= 100
 R3: 10 q_2^4 + 10 I_1^4 - 10 B_1^4 + theta_3^4 >= 100
 R4: 10 q_2^4 + 10 I_1^4 - 10 B_1^4 + theta_3^4 >= 100
 R5: 10 q_2^4 + 10 I_1^4 - 10 B_1^4 + theta_3^4 >= 100
 R6: 4 q_2^4 + 4 I_1^4 - 4 B_1^4 + theta_3^4 >= 52
 R7: 4 q_2^4 + 4 I_1^4 - 4 B_1^4 + theta_3^4 >= 52
 R8: 2 q_2^4 + 2 I_1^4 - 2 B_1^4 + theta_3^4 >= 29
 R9: theta_3^4 >= 0
Bounds
 Constant = 1
End
