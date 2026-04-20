# 🟡 Pacman DQN Agent (Reinforcement Learning)

Proyecto práctico para entender cómo funciona **Deep Reinforcement Learning (DQN)** aplicándolo al entorno de **Pacman (ALE/Pacman-v5)** usando Gymnasium.

Este proyecto es una extensión del trabajo realizado con **LunarLander-v3**, adaptando el enfoque a un entorno mucho más complejo basado en imágenes.

---

# 🎮 Entorno: Pacman (ALE/Pacman-v5)

Se utiliza el entorno de Atari a través de Gymnasium:

👉 https://ale.farama.org/environments/pacman/

Pacman debe recorrer un laberinto:

- Comer todas las bolitas (pellets)
- Evitar los fantasmas
- Usar power pellets para poder comerse fantasmas

---

## 🧠 Estado (State)

A diferencia de LunarLander (vector de 8 valores), aquí el estado es una **imagen**.

Después del preprocesamiento:
# 🟡 Pacman DQN Agent (Reinforcement Learning)

Proyecto práctico para entender cómo funciona **Deep Reinforcement Learning (DQN)** aplicándolo al entorno de **Pacman (ALE/Pacman-v5)** usando Gymnasium.

Este proyecto es una extensión del trabajo realizado con **LunarLander-v3**, adaptando el enfoque a un entorno mucho más complejo basado en imágenes.

---

# 🎮 Entorno: Pacman (ALE/Pacman-v5)

Se utiliza el entorno de Atari a través de Gymnasium:

👉 https://ale.farama.org/environments/pacman/

Pacman debe recorrer un laberinto:

- Comer todas las bolitas (pellets)
- Evitar los fantasmas
- Usar power pellets para poder comerse fantasmas

---

## 🧠 Estado (State)

A diferencia de LunarLander (vector de 8 valores), aquí el estado es una **imagen**.

Después del preprocesamiento:
(84,84,4)

Esto representa:

- 84x84 → imagen en escala de grises
- 4 → últimos 4 frames (Frame Stacking)

👉 Esto permite al agente inferir movimiento (velocidad y dirección).

---

## 🎮 Acciones (Actions)

Las acciones son discretas y dependen del entorno Atari:

| Acción | Descripción |
|------|-------------|
| 0 | No hacer nada |
| 1 | Arriba |
| 2 | Derecha |
| 3 | Izquierda |
| 4 | Abajo |

---

## 🎯 Recompensas (Rewards)

Las recompensas son definidas por el entorno:

| Evento | Recompensa |
|------|-----------|
| Comer pellet | positiva |
| Comer fantasma | alta positiva |
| Perder vida | negativa |
| Avanzar en el juego | pequeña positiva |

⚠️ Importante:

El agente **NO entiende el objetivo del juego**, solo aprende a maximizar la recompensa acumulada.

---

# 🤖 Algoritmo: Deep Q-Network (DQN)

Debido a que el estado es una imagen, no se puede usar Q-table.

Se utiliza una red neuronal (CNN):

Estado (imagen) → CNN → Q-values → Acción


---

## 🔑 Componentes clave

### 1. Experience Replay

- Almacena experiencias pasadas
- Entrena con muestras aleatorias
- Reduce correlación entre datos

---

### 2. Target Network

- Copia congelada de la red principal
- Se actualiza periódicamente
- Estabiliza el entrenamiento

---

### 3. ε-greedy (Exploración vs Explotación)

- ε alto → exploración
- ε bajo → explotación

---

# 🏗️ Arquitectura del modelo (CNN)

Entrada:
(4,84,84)

Salida: Q-values (número de acciones)

La red aprende patrones visuales como:

- posición de Pacman
- fantasmas
- paredes
- pellets

---

# 🧪 Entrenamiento

Ejecutar:

```bash
python train_dqn.py

salida ejemplo:
Episode 30 | Reward: 40.00 | Epsilon: 0.10

📊 RESULTADOS OBTENIDOS
Recompensa promedio: ~20 – 40
Máximo observado: ~90
Comportamiento:
Aprende a moverse
Come pellets
Evita parcialmente peligros

⚠️ Limitaciones
No completa el nivel
No estrategia avanzada contra fantasmas
Reward no está optimizado (sin reward shaping)

📁 ESTRUCTURA DEL PROYECTO
Pacman_game/
│
├── dqn_agent.py        # Agente DQN
├── dqn_cnn.py          # Red neuronal (CNN)
├── wrappers.py         # Preprocesamiento (resize, grayscale, stack)
├── train_dqn.py        # Entrenamiento
├── test_pacman.py      # Prueba del entorno
├── test_cnn.py         # Validación de la red
│
├── results/
│   ├── dqn_pacman_base.png
│   └── dqn_pacman_base.pth
│
└── README.md

💾 GUARDADO DEL MODELO

El modelo entrenado se guarda como:

results/dqn_pacman_base.pth

🚀 TRABAJO FUTURO (mejoras)

Para mejorar el agente:

🔥 Reward Shaping
Penalizar morir
Recompensar supervivencia
Incentivar pellets

🔥 Más entrenamiento
500–1000 episodios

🔥 Mejoras en DQN
Double DQN
Prioritized Replay

🔥 Estrategia avanzada
Evitar fantasmas activamente
Usar power pellets inteligentemente

🧠 CONCLUSIÓN

Este proyecto demuestra:

Implementación completa de DQN desde cero
Adaptación de RL a entornos visuales
Diferencias entre entornos simples y complejos

Aunque el agente aún no domina el juego, logra:

✔ aprender comportamiento básico
✔ mejorar con el tiempo
✔ sentar base para mejoras avanzadas
