# EXPERIENCIAS EN ANALÍTICA

## ENTREGA

**Integrantes:**

Yeniffer Andrea Córdoba

Pablo Gómez Mutis

Luz Adriana Yepes

## DESARROLLO

Para esta entrega se realizó un modelo de **regresión lineal simple**.

En la carpeta de **model**, se agregaron dos nuevos scripts con los códigos de construcción (**build_regresion.py**) y entrenamiento (**training_regresion.py**) del modelo de regresión. En la carpeta de **data**, se agregaron dos nuevos scripts de carga de datos (**load_regresion.py**) y preprocesamiento de datos (**preprocess_regresion.py**). Estos scripts se basaron en aquellos ya existentes en el repositorio, pero adaptados al proyecto de la regresión linear.

Después, en la carpeta de **workflows**, se agregaron nuevos archivos: **build_regresion.yml**, **load_data_regresion.yml**, **preprocess_data_regresion.yml** y **train_regresion.yml**. Estos archivos son muy similares a los originales, pero en los nuevos se cambian los scripts con los que se hace el workflow a los de la regresión linear.

Una vez creados los workflows, volvemos a los scripts en la parte de src y hacemos una pequeña modificación (en este caso, se agregó un comentario: #comment). Luego, se le da click a **commit changes** y vamos a la sección de **actions**, donde veremos que empezó el proceso automatizado de update de los distintos scripts. Nos aseguramos de que todos se ejecuten correctamente.

<img width="1055" alt="Captura de Pantalla 2025-04-25 a la(s) 12 01 14 p m" src="https://github.com/user-attachments/assets/ad36b2b3-7ff7-47ff-be37-38e328d5983a" />

Tras hacer esto, si vamos a la cuenta de Wandb que se ligó al repositorio, podemos ver el nuevo artifact: **Simple Linear Regression**.

<img width="502" alt="Captura de Pantalla 2025-04-25 a la(s) 10 32 22 a m" src="https://github.com/user-attachments/assets/b49a95e4-8bfa-4f84-b2f8-a56abbe826cf"/> 


Tras ejecutar todos los pasos del procedimiento, podremos ver en la interfaz de **lineage** de Wandb el pipeline del proyecto.

<img width="1026" alt="Captura de Pantalla 2025-04-25 a la(s) 12 02 16 p m" src="https://github.com/user-attachments/assets/b839f58a-2c5f-4766-8171-a1004daa51b6" />



