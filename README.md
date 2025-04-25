# EXPERIENCIAS EN ANALÍTICA

## ENTREGA

**Integrantes:**

Yeniffer Andrea Córdoba

Pablo Gómez Mutis

Luz Adriana Yepes

## DESARROLLO

Para esta entrega se realizó un modelo de **regresión lineal simple**.

En la carpeta de **model**, se agregaron dos nuevos scripts con los códigos de construcción (**build_regresion.py**) y entrenamiento (**training_regresion.py**) del modelo de regresión. Estos scripts se hicieron basándose en los scripts originales del modelo de clasificación visto en clase, utilizando los mismos nombres de las variables que se establecieron en la carga y preprocesamiento de los datos.

Después, en la carpeta de **workflows**, se agregaron dos nuevos archivos: **build_regresion.yml** y **train_regresion.yml**. Estos dos archivos automatizan la ejecución de los nuevos scripts que agregamos al repositorio. Estos archivos .yml se hicieron basándose en los que ya había en la carpeta, se adaptaron simplemente los nombres de los files.

Una vez creados los workflows, volvemos a los scripts en la parte de model y hacemos una pequeña modificación (en este caso, se agregó un comentario: #comment). Luego, se le da click a **commit changes** y vamos a la sección de **actions**, donde veremos que empezó el proceso automatizado de update tanto de build_regresion como de train_regresion.

<img width="502" alt="Captura de Pantalla 2025-04-25 a la(s) 10 32 22 a m" src="https://github.com/user-attachments/assets/b49a95e4-8bfa-4f84-b2f8-a56abbe826cf" />

Tras ejecutar todos los pasos del procedimiento, podremos ver en la interfaz de lineage de Wandb el pipeline del proyecto.



