\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  q_2^2 + 2 I_1^2 + 15 B_1^2 + theta_3^2 - 225 Constant
Subject To
 R1: 15 q_2^2 + 15 I_1^2 - 15 B_1^2 + theta_3^2 >= 150
 R3: - 2 q_2^2 - 2 I_1^2 + 2 B_1^2 + theta_3^2 >= -20
 R4: I_1^2 - B_1^2 = -5.454545454545455
Bounds
 theta_3^2 >= -100
 Constant = 1
End
