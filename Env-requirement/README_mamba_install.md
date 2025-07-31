# ✅ Installing Mamba-SSM on GCP or HPC  
📅 **Date: 2025/06/16**

🔗 Official Mamba GitHub: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

---

## 🧩 Step-by-step Installation Instructions

To install `mamba-ssm` successfully and avoid CUDA/nvcc-related issues, follow these 5 steps:

---

### **Step 1: Open the Environment File**

```bash
vim JHMamba2-env20250529.yml
```

### **Step 2: Comment Out These 3 Packages**
These packages should be installed manually later due to potential CUDA build issues:
```bash
# causal-conv1d==1.5.0.post8
# mamba-ssm==2.2.4
# numpy==2.2.0
```

### **Step 3: Create the Conda Environment**
```bash
conda env create -f JHMamba2-env20250529.yml -n mamba2
conda activate mamba2
```

### **Step 4: Install CUDA Toolkit (Needed for nvcc Compiler)**
mamba-ssm requires the NVIDIA CUDA Compiler (nvcc) to build its CUDA core modules.
```bash
conda install -c nvidia cuda-toolkit=12.4
```

### ** Step 5: Manually Install Mamba-SSM (with causal-conv1d)**
```bash
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
```

## 📌 Why this step matters:
- The official release (v2.2.4) has a bug: bare_metal_version is undefined if nvcc is not available.
- Using --no-build-isolation avoids pip internally rebuilding packages with conflicting or incomplete dependencies.
- Installing both mamba-ssm and causal-conv1d together ensures compatibility with Torch 2.2+ and custom CUDA kernels.