# 获取当前目录
$HGCN_HOME = Get-Location
$env:HGCN_HOME = $HGCN_HOME

# 设置日志和数据路径
$env:LOG_DIR = "$env:HGCN_HOME\logs"
$env:DATAPATH = "$env:HGCN_HOME\data"

# 设置 PYTHONPATH (Windows 使用分号 ; 分隔)
$env:PYTHONPATH = "$env:HGCN_HOME;$env:PYTHONPATH"

# Windows 下没有 LD_LIBRARY_PATH，通常是将 CUDA 的 bin 目录加入 PATH
# 假设你的 CUDA 安装在默认路径
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp;$env:PATH"

# 激活环境 (如果是 Conda)
# conda activate hgcn
