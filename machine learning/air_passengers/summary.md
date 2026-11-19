- 没想到最终的预测效果反而是指数平滑最好，其次是 arima，然后是 lstm，最后是 rnn
- lstm 的优势估计在于长序列预测上（超过1000个时间步）

## Mac

LSTM predict the next 12 months:

测试集 MAE: 21.1955

测试集 MAPE: 4.88%

测试集 RMSE: 25.7138

测试集 MASE: 0.7205

Seasonal Naive MASE: 1.5900

LSTM predict the next 1 month:

测试集 MAE: 23.8772
测试集 MAPE: 5.55%
测试集 RMSE: 28.5328
测试集 MASE: 0.8117
Seasonal Naive MASE: 1.4043

Exponential Smoothing:
MAE  : 17.4627
MAPE : 4.2565%
RMSE : 21.3371
MASE : 0.5936
Seasonal Naive MASE: 1.4043

Seasonal Arima:
MAE  : 18.6288
MAPE : 4.1505%
MASE : 0.6333
RMSE : 23.4691
Seasonal Naive MASE: 1.4043