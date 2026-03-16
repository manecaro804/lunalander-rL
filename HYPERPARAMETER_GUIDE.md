python quick_test.py --agent both --episodes 300 --plot# Guía de Búsqueda de Hiperparámetros

Esta guía te ayudará a encontrar la mejor combinación de hiperparámetros para tus modelos de RL.

## 📋 Resumen de Hiperparámetros

### Q-Learning
- **n_bins**: Número de bins para discretizar el espacio de observación (5-20). Mayor = más precisión pero más estados.
- **lr** (learning rate): Tasa de aprendizaje (0.05-0.2). Mayor = aprendizaje más rápido pero menos estable.
- **gamma**: Factor de descuento (0.95-0.999). Peso del futuro vs inmediato.
- **epsilon_decay**: Decaimiento de epsilon (0.99-0.999). Cómo reduce la exploración.

### DQN
- **lr**: Tasa de aprendizaje (1e-4 a 1e-2). Afecta velocidad de convergencia.
- **gamma**: Factor de descuento (0.9-0.999). Importancia de recompensas futuras.
- **epsilon_decay**: Decaimiento de epsilon (0.99-0.9999). Transición exploración → explotación.
- **batch_size**: Tamaño del mini-batch (32-128). Balance entre estabilidad y velocidad.
- **hidden**: Número de neuronas(64-256). Capacidad del modelo.
- **target_update_freq**: Cada cuántos episodios actualizar red objetivo (5-50). Mayor = más estabilidad.

## 🚀 Opciones de Uso

### Opción 1: Pruebas Rápidas (RECOMENDADO PARA EMPEZAR)

Prueba configuraciones predefinidas rápidamente:

```bash
# Comparar varias configuraciones de Q-Learning
python quick_test.py --agent qlearning --episodes 500

# Comparar varias configuraciones de DQN
python quick_test.py --agent dqn --episodes 350

# Probar ambos agentes
python quick_test.py --agent both --episodes 500

# Probar solo una configuración específica
python quick_test.py --agent qlearning --config baseline --episodes 500

# Generar gráficos de comparación
python quick_test.py --agent both --episodes 500 --plot
```

**Ventajas:**
- Rápido (10-30 minutos)
- Fácil de interpretar
- Resultados visuales

### Opción 2: Búsqueda Exhaustiva (GRID SEARCH)

Prueba TODAS las combinaciones de Q-Learning:

```bash
# Búsqueda completa (puede tomar 1-2 horas)
python hyperparameter_tuning.py --agent qlearning --episodes 500 --runs 2

# Con configuración personalizada
python hyperparameter_tuning.py --agent qlearning --episodes 800 --runs 3
```

**Ventajas:**
- Encuentra los óptimos locales
- Exhaustivo y confiable

**Desventajas:**
- Muy lento

### Opción 3: Optimización Bayesiana (RECOMENDADO PARA DQN)

Búsqueda inteligente de hiperparámetros para DQN:

```bash
# Optimización Bayesiana (30-60 minutos, 20 pruebas)
python hyperparameter_tuning.py --agent dqn --episodes 200 --trials 20

# Más pruebas = mejor resultado pero más lento
python hyperparameter_tuning.py --agent dqn --episodes 300 --trials 50
```

**Ventajas:**
- Inteligente y eficiente
- Mejor que búsqueda aleatoria
- Explora bien el espacio

## 📊 Entendiendo los Resultados

Los scripts generan resultados JSON en la carpeta `tuning_results/` o `quick_tests/`:

```
{
  "params": {
    "lr": 0.1,
    "gamma": 0.99,
    ...
  },
  "avg_final_reward": -150.45,
  "std_final_reward": 25.34,
  "run_results": [...]
}
```

**Métricas importantes:**
- **final_avg_reward**: Promedio de recompensa en últimos episodios (entre más alto, mejor)
- **std_final_reward**: Desviación estándar (entre más baja, más estable)
- **max_reward**: Máxima recompensa observada

## 💡 Consejos Prácticos

### Para Q-Learning:
1. **Empieza con n_bins=10**: Buen balance entre discretización y complejidad
2. **Prueba epsilon_decay entre 0.99-0.995**: Afecta exploración vs explotación
3. **lr=0.1 suele funcionar bien**: Ajusta si converge muy rápido/lento
4. **gamma=0.99**: Típicamente predeterminado bueno

### Para DQN:
1. **Empieza con hidden=128**: Red pequeña pero potente
2. **lr=1e-3**: Buen punto de partida, reduce si es inestable
3. **batch_size=64**: Balance sólido entre estabilidad y velocidad
4. **target_update_freq=10**: Cada 10 episodios actualiza red objetivo

### Estrategia Recomendada:

1. **Fase 1 (5-10 min):** Corre un test rápido con configuraciones predefinidas
   ```bash
   python quick_test.py --agent both --episodes 300
   ```

2. **Fase 2 (20-60 min):** Refina el mejor agente
   - Para Q-Learning: Grid search con rango más pequeño
   - Para DQN: Optimización Bayesiana con 20-30 trials

3. **Fase 3 (opcional):** Entrenamiento final con mejores hiperparámetros
   ```bash
   # Entrena una vez más con los mejores parámetros
   rlgames train --agent dqn --episodes 2000
   ```

## 🔍 Analizando Diferentes Hiperparámetros

### Exploración vs Explotación
- **epsilon_decay alto (0.999)**: Explora mucho tiempo → mejor solución final pero lento
- **epsilon_decay bajo (0.99)**: Converge rápido pero puede ser subóptimo

### Capacidad del Modelo (DQN)
- **hidden=64**: Rápido pero puede ser limitado
- **hidden=128**: Balance típico
- **hidden=256**: Poderoso pero más lento

### Tasa de Aprendizaje
- **lr muy alto**: Inestable, oscila alrededor de la solución
- **lr muy bajo**: Muy lento, puede no converger
- **lr óptimo**: Converge suave y rápido

## 📈 Comparando Resultados

Los gráficos generados muestran claramente:
- Un agente que converge rápido tiene curva empinada temprano
- Un agente estable tiene menos fluctuaciones
- Un agente óptimo combina convergencia rápida + estabilidad

## ⚙️ Requisitos Previos

```bash
# Instala las dependencias necesarias
pip install optuna matplotlib
```

## 🎯 Tus Próximos Pasos

1. **Hoy**: Corre `python quick_test.py --agent both --episodes 300`
2. **Mañana**: Refina con búsqueda exhaustiva o Bayesiana
3. **Después**: Entrena el modelo final con los mejores parámetros

¡Buena suerte! 🚀
