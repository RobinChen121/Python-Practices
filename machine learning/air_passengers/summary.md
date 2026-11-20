- 没想到最终的预测效果反而是指数平滑最好，其次是 arima，然后是 lstm，最后是 rnn
- lstm 的优势估计在于长序列预测上（超过1000个时间步）

MAC 和 Windows 有浮点运算差异

## LSTM

#### LSTM predict the next 12 months:

Windows:

测试集 MAE: 21.1955

测试集 MAPE: 4.88%

测试集 RMSE: 25.7138

测试集 MASE: 0.7205

Seasonal Naive MASE: 1.5900

### predict the next 1 month

测试集 MAE: 23.8772
测试集 MAPE: 5.55%
测试集 RMSE: 28.5328
测试集 MASE: 0.8117
Seasonal Naive MASE: 1.4043

Windows:
测试集 MAE: 22.4515
测试集 MAPE: 5.31%
测试集 RMSE: 27.1271
测试集 MASE: 0.7632
Seasonal Naive MASE: 2.2014

## Exponential Smoothing

MAC:

MAE  : 17.4665
MAPE : 4.2574%
RMSE : 21.3402
MASE : 0.5937
Seasonal Naive MASE: 2.2014

Windows:
MAE  : 17.4665
MAPE : 4.2574%
RMSE : 21.3402
MASE : 0.5937
Seasonal Naive MASE: 2.2014

## Seasonal Arima

MAC:

MAE  : 18.6288
MAPE : 4.1505%
MASE : 0.6333
RMSE : 23.4691
Seasonal Naive MASE: 1.4043

Windows:

MAE  : 18.6289
MAPE : 4.1505%
RMSE : 23.4692
MASE : 0.6333
Seasonal Naive MASE: 2.2014