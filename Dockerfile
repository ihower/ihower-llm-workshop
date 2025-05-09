FROM python:3.10-slim

# 創建一個非 root 用戶
RUN groupadd -r sandboxuser && useradd -r -g sandboxuser sandboxuser

WORKDIR /home/sandboxuser

# 安裝常用資料分析和機器學習套件
RUN pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn

# 安裝其他常用套件
RUN pip install --no-cache-dir requests jupyter plotly

# 創建工作目錄並設置權限
RUN mkdir -p /home/sandboxuser/data && \
    chown -R sandboxuser:sandboxuser /home/sandboxuser

# 切換到非 root 用戶
USER sandboxuser

# 容器運行時的命令
CMD ["bash"]