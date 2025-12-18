# PyTorch Profiler: CPU & GPU Performance Analysis

Este repositorio muestra **cómo usar `torch.profiler` de forma práctica y progresiva**
para analizar **tiempo, memoria y ejecución** de un modelo de deep learning en PyTorch.

El ejemplo utiliza **ResNet-18** con datos sintéticos y cubre:
- Profiling en CPU
- Profiling en GPU (CUDA / XPU)
- Uso de memoria
- Exportación de trazas
- Stack traces
- Schedules de profiling

---

## 🧠 ¿Qué se aprende en este repo?

✔️ Cómo perfilar inferencia de un modelo en CPU  
✔️ Cómo identificar cuellos de botella en GPU  
✔️ Diferencia entre tiempo real de cómputo y sincronización  
✔️ Cómo analizar uso de memoria  
✔️ Cómo generar trazas visuales (`trace.json`)  
✔️ Cómo usar `schedule` para profiling en loops reales  

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
