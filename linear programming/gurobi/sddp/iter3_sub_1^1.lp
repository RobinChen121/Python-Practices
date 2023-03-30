\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^1 + 2 I_1^1 + 15 B_1^1 + theta_3^1 - 75 Constant
Subject To
 R1: 15 q_2^1 + 15 I_1^1 - 15 B_1^1 + theta_3^1 >= 150
 R3: - 2 q_2^1 - 2 I_1^1 + 2 B_1^1 + theta_3^1 >= -20
 R4: I_1^1 - B_1^1 = 4.545454545454545
Bounds
 theta_3^1 >= -100
 Constant = 1
End
