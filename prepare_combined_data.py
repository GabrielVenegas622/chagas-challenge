# prepare_combined_data.py
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import random # Todavía útil para el shuffle final si quieres


# --- Configuration ---
# Rutas a tus carpetas de salida preprocesadas
# Asegúrate de que estas rutas sean correctas en tu sistema.
# Asumo que este script se ejecuta desde la carpeta 'randomforest'
# y que 'samitrop_output' y 'ptbxl_output' están un nivel arriba.
SAMITROP_INPUT_DIR = '../samitrop_output'
PTBXL_INPUT_DIR = '../ptbxl_output' 

# Directorios de salida para datos de entrenamiento y holdout
TRAIN_OUTPUT_DIR = '../training_data'
HOLDOUT_OUTPUT_DIR = '../holdout_data'

# Proporción para el split (0.2 para 20% holdout, 80% training)
TEST_SIZE = 0.2
RANDOM_SEED = 42 # Para reproducibilidad

# --- Crear directorios de salida si no existen ---
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(HOLDOUT_OUTPUT_DIR, exist_ok=True)

print(f"Leyendo archivos de Samitrop desde: {SAMITROP_INPUT_DIR}")
print(f"Leyendo archivos de PTB-XL desde: {PTBXL_INPUT_DIR}")
print(f"Los datos de entrenamiento irán a: {TRAIN_OUTPUT_DIR}")
print(f"Los datos de holdout irán a: {HOLDOUT_OUTPUT_DIR}")

# --- Paso 1: Recolectar todos los IDs y asignar etiquetas ---
all_record_ids = [] # Lista para almacenar solo los IDs (e.g., 'ID123')
all_labels = []     # Lista para almacenar las etiquetas (True/False)
record_source_paths = {} # Diccionario para mapear ID -> Ruta original de la carpeta (para copiar después)

# Procesar Samitrop (asumiendo que todos son positivos de Chagas)
print("\nProcesando datos de Samitrop (asumiendo todos son positivos)...")
samitrop_ids = []
for filename in os.listdir(SAMITROP_INPUT_DIR):
    if filename.endswith('.dat'):
        record_id = os.path.splitext(filename)[0] # Extraer el ID sin extensión
        samitrop_ids.append(record_id)
        record_source_paths[record_id] = SAMITROP_INPUT_DIR # Guardar la ruta de origen

        all_record_ids.append(record_id)
        all_labels.append(True) # Chagas positivo

if not samitrop_ids:
    print(f"Advertencia: No se encontraron archivos .dat en {SAMITROP_INPUT_DIR}. Por favor, verifica la ruta.")
else:
    print(f"Encontrados {len(samitrop_ids)} registros de Samitrop (positivos).")

# Procesar PTB-XL (asumiendo que todos son negativos de Chagas)
print("\nProcesando datos de PTB-XL (asumiendo todos son negativos)...")
ptbxl_ids = []
# Iterar a través de las subcarpetas de PTB-XL (00000, 01000, etc.)
for root, dirs, files in os.walk(PTBXL_INPUT_DIR):
    for filename in files:
        if filename.endswith('.dat'):
            record_id = os.path.splitext(filename)[0] # Extraer el ID
            ptbxl_ids.append(record_id)
            record_source_paths[record_id] = root # Guardar la subcarpeta como ruta de origen
            
            all_record_ids.append(record_id)
            all_labels.append(False) # Chagas negativo

if not ptbxl_ids:
    print(f"Advertencia: No se encontraron archivos .dat en {PTBXL_INPUT_DIR} o sus subdirectorios. Por favor, verifica la ruta.")
else:
    print(f"Encontrados {len(ptbxl_ids)} registros de PTB-XL (negativos).")

if not all_record_ids:
    print("Error: No se encontraron registros en ningún directorio de entrada. Saliendo.")
    exit()

print(f"\nTotal de registros únicos encontrados: {len(all_record_ids)}")
print(f"Total de registros positivos: {sum(all_labels)}")
print(f"Total de registros negativos: {len(all_labels) - sum(all_labels)}")

# Convertir a arrays de numpy para scikit-learn
all_record_ids_np = np.array(all_record_ids)
all_labels_np = np.array(all_labels)

# --- Paso 2: Realizar el split estratificado (80-20) ---
# train_ids_raw: Contiene los IDs de los registros para el conjunto de entrenamiento (antes de undersampling)
# holdout_ids: Contiene los IDs de los registros para el conjunto de holdout
print("\nRealizando split estratificado train-holdout (80-20)...")
train_ids_raw, holdout_ids, train_labels_raw, holdout_labels_raw = train_test_split(
    all_record_ids_np, # Los IDs de los registros
    all_labels_np,     # Las etiquetas correspondientes a esos IDs
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=all_labels_np # CRUCIAL: para mantener la proporción de clases en train y holdout
)

print(f"Tamaño del conjunto de entrenamiento 'crudo': {len(train_ids_raw)} registros.")
print(f"Tamaño del conjunto de holdout: {len(holdout_ids)} registros.")

# --- Paso 3: Aplicar Undersampling SOLO al conjunto de entrenamiento usando imbalance-learn ---
print("\nAplicando undersampling al conjunto de entrenamiento con imblearn...")

# Contar clases antes del undersampling para el set de entrenamiento
num_positive_train_raw = np.sum(train_labels_raw == True)
num_negative_train_raw = np.sum(train_labels_raw == False)

print(f"Conjunto de entrenamiento (antes de undersampling) - Positivos: {num_positive_train_raw}")
print(f"Conjunto de entrenamiento (antes de undersampling) - Negativos: {num_negative_train_raw}")

# Inicializar RandomUnderSampler
# sampling_strategy='majority' eliminará muestras de la clase mayoritaria
# random_state para reproducibilidad
rus = RandomUnderSampler(sampling_strategy='majority', random_state=RANDOM_SEED)

# El undersampler opera en los índices. Creamos un "dummy" array de características
# solo para que el sampler tenga algo sobre lo que operar, pero lo que nos interesa son los índices resultantes.
# reshape(-1, 1) es necesario porque rus.fit_resample espera un 2D array para X.
dummy_X_train = np.arange(len(train_ids_raw)).reshape(-1, 1) # Un array de índices de 0 a N-1
X_resampled_indices, y_resampled = rus.fit_resample(dummy_X_train, train_labels_raw)

# Extraer los IDs de los registros undersampleados
train_ids_final = train_ids_raw[X_resampled_indices.flatten()] # Usar los índices para seleccionar IDs

num_positive_train_final = np.sum(y_resampled == True)
num_negative_train_final = np.sum(y_resampled == False)
print(f"Conjunto de entrenamiento (después de undersampling) - Positivos: {num_positive_train_final}")
print(f"Conjunto de entrenamiento (después de undersampling) - Negativos: {num_negative_train_final}")
print(f"Tamaño final del conjunto de entrenamiento: {len(train_ids_final)} registros.")

# Opcional: mezclar los IDs finales del entrenamiento para que no queden agrupados por clase
random.shuffle(train_ids_final)

# --- Paso 4: Copiar archivos a sus respectivos directorios ---
print("\nCopiando archivos a los directorios 'training_data' y 'holdout_data'...")

# Copiar archivos de entrenamiento
for record_id in train_ids_final:
    source_dir = record_source_paths[record_id] # Usar el path guardado al inicio (Samitrop o subcarpeta PTB-XL)
    for ext in ['.dat', '.hea']:
        src_path = os.path.join(source_dir, f"{record_id}{ext}")
        dst_path = os.path.join(TRAIN_OUTPUT_DIR, f"{record_id}{ext}")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Advertencia: Archivo no encontrado para copiar a entrenamiento: {src_path}")

# Copiar archivos de holdout
for record_id in holdout_ids:
    source_dir = record_source_paths[record_id] # Usar el path guardado al inicio
    for ext in ['.dat', '.hea']:
        src_path = os.path.join(source_dir, f"{record_id}{ext}")
        dst_path = os.path.join(HOLDOUT_OUTPUT_DIR, f"{record_id}{ext}")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Advertencia: Archivo no encontrado para copiar a holdout: {src_path}")

print("\nPreparación y división de datos combinados completada.")
