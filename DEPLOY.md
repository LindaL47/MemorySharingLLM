# 阿里云服务器部署与训练指南

本指南适用于将 MemorySharingLLM 项目部署到 Linux 服务器（如阿里云 ECS，Ubuntu/CentOS 系统）。

## 1. 性能预期分析 (重要)

您提到的配置是 **4核 vCPU / 8GB 内存**。

*   **会更快吗？**
    *   **网络请求 (API Calls)**: **会更快且更稳定**。您的代码频繁调用 SiliconFlow API，云服务器通常拥有更好的骨干网连接，能有效减少 `ConnectionResetError` 和超时重试，这会显著提升整体运行速度。
    *   **模型训练 (BERT Training)**: **可能不会变快，甚至变慢**。
        *   如果您本地电脑有独立显卡 (GPU)，本地训练通常比 4核 CPU 快得多。
        *   如果您本地也是用 CPU 跑，云服务器的 4核 CPU 性能可能与普通笔记本相当或略弱。
        *   **8GB 内存风险**: 运行 BERT 模型 + 数据处理比较吃内存，8GB 比较紧张，建议增加 Swap 交换空间以防内存溢出 (OOM)。

*   **建议**: 如果主要瓶颈是网络报错（如您之前遇到的连接重置），上云会有很大帮助。如果是计算太慢，建议购买带 GPU (如 NVIDIA T4/A10) 的实例，或者优化代码（减少重复加载模型的次数）。

---

## 2. 部署步骤

假设您使用的是 **Ubuntu 20.04/22.04** 系统。

### 第一步：连接服务器并初始化环境

1.  使用 SSH 连接服务器：
    ```bash
    ssh root@<您的公网IP>
    ```

2.  更新系统并安装基础工具：
    ```bash
    apt-get update
    apt-get install -y git python3-pip python3-venv screen
    ```

### 第二步：上传代码

您可以使用 `scp` 命令在本地终端将代码上传到服务器（或者使用 Git）。

**本地终端执行 (不是在服务器上):**
```bash
# 假设您的项目在 d:\APP\Cursor\MemorySharingLLM\MemorySharingLLM
# 上传整个文件夹到服务器的 /root/project 目录
scp -r "d:\APP\Cursor\MemorySharingLLM\MemorySharingLLM" root@<您的公网IP>:/root/project
```

### 第三步：配置 Python 环境

回到服务器终端执行：

1.  进入项目目录：
    ```bash
    cd /root/project
    ```

2.  创建虚拟环境（推荐，避免污染系统环境）：
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  安装依赖：
    ```bash
    # 升级 pip
    pip install --upgrade pip
    
    # 安装项目依赖 (CPU版 PyTorch 比较小，如果服务器没 GPU 建议指定安装 CPU 版)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```

4.  设置 Swap (虚拟内存) - **强烈建议 8GB 内存机器执行此步**：
    ```bash
    # 创建 4GB 的 Swap 文件
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    # 永久生效
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
    ```

### 第四步：运行训练

由于训练时间较长，建议使用 `screen` 或 `nohup` 后台运行，这样断开 SSH 连接后程序不会停止。

**使用 Screen (推荐):**

1.  创建一个新窗口：
    ```bash
    screen -S train_task
    ```

2.  运行代码 (以 Literal Creation 为例)：
    ```bash
    # 确保在虚拟环境中
    source venv/bin/activate
    
    # 进入目录
    cd "Literal Creation"
    
    # 运行
    python Integrate.py
    ```

3.  **挂起后台**: 按键盘 `Ctrl + A`，然后按 `D`。此时你可以安全关闭终端。

4.  **查看进度**: 重新登录后，执行 `screen -r train_task` 恢复窗口。

---

## 3. 代码优化建议 (针对速度)

目前代码中有一个严重的性能瓶颈：**在循环中重复加载模型**。

在 `Agent_MS.py` 的 `train_model_duringStore` 函数中，每次存储一条记忆都会重新加载模型、训练、再保存模型。
```python
# 每次调用都发生 IO 操作
Retrival_model.load_model_and_optimizer(...) 
# ... 训练 ...
Retrival_model.save_model_and_optimizer(...)
```
如果在云服务器上依然觉得慢，建议后续优化逻辑：将模型加载放在循环外，只在内存中更新权重，每隔 N 次循环再保存一次到硬盘。
