"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/7/9 16:22
Description: 
    

"""
import streamlit as st

st.title("在线计算器")

a = st.number_input("输入第一个数字", value=0.0)
b = st.number_input("输入第二个数字", value=0.0)

if st.button("计算加法"):
    st.success(f"结果是：{a + b}")