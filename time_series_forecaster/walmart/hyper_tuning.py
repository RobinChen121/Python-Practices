"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/10 12:46
Description: 


"""
from lstm_model import run_lstm
import optuna
import torch
from utils.get_pc_info import get_info

class HyperTuning(object):
    def __init__(self, X, y, scaler, file_name, best_loss):
        self.X = X
        self.y = y
        self.file_name = file_name
        self.scaler = scaler
        self.best_loss = best_loss


    def objective(self, trial):
        # 超参数采样
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 3)  # sample from 1 to 3
        # dropoutΩpout = trial.suggest_float('dropout', 0.0, 0.5)
        bidirectional = trial.suggest_categorical("bidirectional", [False])
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )  # 1e-4 到 1e-2 的范围内用对数采样学习率
        batch_size = trial.suggest_categorical("batch_size", [1])

        loss, _, _ = run_lstm(
            self.X, self.y, self.scaler, self.file_name, self.best_loss, batch_size, hidden_size, num_layers, bidirectional, learning_rate
        )
        return loss


    def tune(self, tune_trial):
        # 创建优化器
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=tune_trial)  # 20次尝试

        print("best hyperparameters:", study.best_params)

        # 保存 checkpoint
        if study.best_value < self.best_loss:
            self.best_loss = study.best_value
            torch.save({
                "best_MAE": study.best_value,
                "hidden_size": study.best_params["hidden_size"],
                "num_layers": study.best_params["num_layers"],
                "bidirectional": study.best_params["bidirectional"],
                "learning_rate": study.best_params["learning_rate"],
                "batch_size": study.best_params["batch_size"],
            }, self.file_name + "_hyperparameter" + ".pth")

        text_name = self.file_name + "_pc" + ".txt"
        with open(text_name, "a") as f:
            import platform
            import time

            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"date: {formatted_time}\n")
            f.write(f"file name: {self.file_name}\n")

            info = get_info()
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.is_available()}\n")
            else:
                f.write(f"GPU: {torch.backends.mps.is_available()}\n")
            f.write(f"Python version: {platform.python_version()}\n")
            f.write(f"best hyperparameters:\n{str(study.best_params)}\n")
            f.write(f"real MAE: {str(study.best_value)}\n")
            f.write(f"historical best MAE: {str(self.best_loss)}\n")
            equals = "==" * 50
            f.write(f"{equals}\n")
        return study.best_params