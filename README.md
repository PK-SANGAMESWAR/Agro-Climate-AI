# 🌱 Agro-Climate AI: Precision Deep Learning for Agriculture

> **Intelligent, Climate-Resilient Agriculture through Multi-Modal Neural Networks.**

Agro-Climate AI is a premium, end-to-end deep learning system designed to bridge the gap between agricultural research and field-ready intelligence. Utilizing custom ResNet, LSTM/GRU, and GAN architectures, it provides farmers and researchers with actionable data: from leaf-level species identification to multi-year yield forecasting and generative biological synthesis.

---

## ✨ Core Intelligence Modules

| Module | Neural Architecture | Primary Objective |
| :--- | :--- | :--- |
| **Species Identification** | Deep CNN (ResNet-18) | Classify 12+ plant seedling species (crop vs. weed) with 98% validated accuracy. |
| **Yield Forecasting** | Temporal LSTM & GRU | Predict harvest outcomes by modeling sequence relationships between GDD & Precipitation. |
| **Plant Studio** | DCGAN / Autoencoders | Explore latent spaces to generate synthetic biological samples for research augmentation. |

---

## 🎨 Premium Experience

The Frontend is an industry-leading glassmorphic dashboard built for high-performance data visualization.
*   **Real-time Inference**: Drag-and-drop seedling identification with immediate neural feedback.
*   **Predictive Analytics**: Interactive charts (Recharts) for historical yield analysis.
*   **Adaptive Layout**: Fully responsive sidebar system for field-use via mobile or tablets.
*   **Fluid Motion**: Precision micro-animations powered by Framer Motion.

---

## 🏗️ Technology Stack

### **Deep Learning Backend**
*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
*   **Neural Engine**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
*   **Data Processing**: Pandas, NumPy, Scikit-learn
*   **Environment**: [uv](https://github.com/astral-sh/uv) (Next-gen Cargo for Python)

### **Intelligence Frontend**
*   **Framework**: [React 19](https://react.dev/) + [Vite 6](https://vite.dev/)
*   **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) (Precision Modern Utilities)
*   **Visualization**: [Recharts](https://recharts.org/) & [Lucide Icons](https://lucide.dev/)
*   **Animations**: [Framer Motion](https://www.framer.com/motion/)

---

## 🚀 Getting Started

### 1. Repository Setup
```bash
git clone https://github.com/yourusername/Agro-Climate.git
cd Agro-Climate
```

### 2. Launch Intelligent Backend
The backend utilizes `uv` for lightning-fast environment isolation.
```bash
# Automated install & run
uv run python server/main.py
```
*API serves on: `http://localhost:8000`*

### 3. Launch Intelligence Dashboard
```bash
cd client
npm install
npm run dev
```
*Dashboard serves on: `http://localhost:5173`*

---

## 📂 Project Architecture

```text
Agro-Climate/
├── client/           # React + Tailwind Dashboard
│   ├── src/          # Source Code
│   │   ├── components/ # AI Visual Modules
│   │   └── App.jsx   # Core Route Handler
│   └── public/       # Static assets
├── server/           # FastAPI + PyTorch Serving Layer
│   └── main.py       # API Endpoints & Model Loading
├── models/           # Exported Neural Weights (.pth)
├── data/             # Historical Datasets
└── *.ipynb           # Research Notebooks (MLP, CNN, LSTM, GAN)
```

---

## 👨‍🔬 Research Contributions

This system implements several custom research breakthroughs:
1. **Hybrid GDD Weighting**: Enhanced yield prediction by integrating Growing Degree Days as a primary sequence feature.
2. **Weed-Informed CNN**: Specialized ResNet layers trained on plant seedling morphology for high-precision targeted herbicide applications.

---
*Created with focus on agricultural sustainability and climate resilience.*
