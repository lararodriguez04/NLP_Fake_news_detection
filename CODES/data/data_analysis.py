import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ============================================
# 1. CARGAR LOS DATOS
# ============================================

# Columnas originales completas
columnas_completas = [
    'claim_id', 'claim', 'date_published', 'explanation',
    'fact_checkers', 'main_text', 'sources', 'label', 'subjects'
]

# Columnas que nos interesan
columnas_interes = ['claim_id', 'claim', 'explanation', 'main_text', 'label']

def cargar_datos(ruta_tsv, nombre_split):
    """Carga un archivo TSV y devuelve DataFrame con columnas de interés"""
    try:
        df = pd.read_csv(ruta_tsv, sep='\t', names=columnas_completas, skiprows=1)
        df = df[columnas_interes]
        df['split'] = nombre_split  # Añadir columna para identificar el split
        print(f"✅ Cargado {nombre_split}: {len(df)} muestras")
        return df
    except FileNotFoundError:
        print(f"❌ No se encontró: {ruta_tsv}")
        return None

def contar_palabras(texto):
    """Cuenta el número de palabras en un texto"""
    if pd.isna(texto) or texto == '':
        return 0
    return len(str(texto).split())

# Rutas de tus archivos
base_path = '/home/lnlpG08/nlp/DATA/'

df_train = cargar_datos(base_path + 'train.tsv', 'train')
df_dev = cargar_datos(base_path + 'dev.tsv', 'dev')
df_test = cargar_datos(base_path + 'test.tsv', 'test')

# Combinar todos para estadísticas globales
df_all = pd.concat([df for df in [df_train, df_dev, df_test] if df is not None], ignore_index=True)

# ============================================
# 2. INFORMACIÓN GENERAL DEL DATASET
# ============================================

print("\n" + "="*80)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*80)

print(f"\n📊 Distribución de muestras por split:")
print(f"   Train: {len(df_train)} muestras")
print(f"   Dev:   {len(df_dev)} muestras")
print(f"   Test:  {len(df_test)} muestras")
print(f"   Total: {len(df_all)} muestras")


# ============================================
# 3. ESTADÍSTICAS POR ETIQUETA (EN PALABRAS)
# ============================================

# Para train
df_train['claim_words'] = df_train['claim'].apply(contar_palabras)
df_train['explanation_words'] = df_train['explanation'].apply(contar_palabras)

# Para dev
df_dev['claim_words'] = df_dev['claim'].apply(contar_palabras)
df_dev['explanation_words'] = df_dev['explanation'].apply(contar_palabras)

# Para test
df_test['claim_words'] = df_test['claim'].apply(contar_palabras)
df_test['explanation_words'] = df_test['explanation'].apply(contar_palabras)

# ============================================
# 4. MOSTRAR ESTADÍSTICAS
# ============================================

print("="*60)
print("ESTADÍSTICAS DE PALABRAS - CLAIM")
print("="*60)

for nombre, df in [('TRAIN', df_train), ('DEV', df_dev), ('TEST', df_test)]:
    print(f"\n📌 {nombre}:")
    print(f"   Media:     {df['claim_words'].mean():.1f} palabras")
    print(f"   Mediana:   {df['claim_words'].median():.0f} palabras")
    print(f"   Mínimo:    {df['claim_words'].min()} palabras")
    print(f"   Máximo:    {df['claim_words'].max()} palabras")
    print(f"   Desv.std:  {df['claim_words'].std():.1f}")

print("\n" + "="*60)
print("ESTADÍSTICAS DE PALABRAS - EXPLANATION")
print("="*60)

for nombre, df in [('TRAIN', df_train), ('DEV', df_dev), ('TEST', df_test)]:
    print(f"\n📌 {nombre}:")
    print(f"   Media:     {df['explanation_words'].mean():.1f} palabras")
    print(f"   Mediana:   {df['explanation_words'].median():.0f} palabras")
    print(f"   Mínimo:    {df['explanation_words'].min()} palabras")
    print(f"   Máximo:    {df['explanation_words'].max()} palabras")
    print(f"   Desv.std:  {df['explanation_words'].std():.1f}")

# ============================================
# 5. TABLA RESUMEN COMPARATIVA
# ============================================

print("\n" + "="*60)
print("TABLA RESUMEN COMPARATIVA")
print("="*60)

resumen = pd.DataFrame({
    'Split': ['TRAIN', 'DEV', 'TEST'],
    'Claim_media': [df_train['claim_words'].mean(), df_dev['claim_words'].mean(), df_test['claim_words'].mean()],
    'Claim_mediana': [df_train['claim_words'].median(), df_dev['claim_words'].median(), df_test['claim_words'].median()],
    'Claim_max': [df_train['claim_words'].max(), df_dev['claim_words'].max(), df_test['claim_words'].max()],
    'Explanation_media': [df_train['explanation_words'].mean(), df_dev['explanation_words'].mean(), df_test['explanation_words'].mean()],
    'Explanation_mediana': [df_train['explanation_words'].median(), df_dev['explanation_words'].median(), df_test['explanation_words'].median()],
    'Explanation_max': [df_train['explanation_words'].max(), df_dev['explanation_words'].max(), df_test['explanation_words'].max()]
})

print(resumen.to_string(index=False, float_format='%.1f'))

# ============================================
# 4. EJEMPLOS POR TIPO DE ETIQUETA
# ============================================

print("\n" + "="*80)
print("EJEMPLOS DE CADA TIPO DE ETIQUETA (del split TRAIN)")
print("="*80)

for label in ['true', 'false', 'mixture', 'unproven']:
    ejemplos = df_train[df_train['label'] == label]['claim'].head(2)
    print(f"\n🔹 {label.upper()}:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"   {i}. {ejemplo[:150]}...")

# ============================================
# 5. VALORES NULOS
# ============================================

print("\n" + "="*80)
print("VALORES NULOS POR COLUMNA")
print("="*80)

def mostrar_nulos(df, nombre):
    if df is not None:
        nulos = df.isnull().sum()
        total = len(df)
        print(f"\n📌 {nombre.upper()}:")
        for col in df.columns:
            if col != 'split':
                count = nulos[col]
                if count > 0:
                    print(f"   {col:15}: {count:5} ({count/total*100:.2f}%)")
                else:
                    print(f"   {col:15}: {count:5} (0.00%)")

mostrar_nulos(df_train, 'train')
mostrar_nulos(df_dev, 'dev')
mostrar_nulos(df_test, 'test')

# ============================================
# 2. CONTAR LABELS EN CADA SPLIT
# ============================================

print("="*60)
print("DISTRIBUCIÓN DE ETIQUETAS (LABELS)")
print("="*60)

# Train
print("\n📌 TRAIN:")
conteo_train = df_train['label'].value_counts()
porcentaje_train = df_train['label'].value_counts(normalize=True) * 100
for label in conteo_train.index:
    print(f"   {label:12}: {conteo_train[label]:5} muestras ({porcentaje_train[label]:5.2f}%)")

# Dev
print("\n📌 DEV:")
conteo_dev = df_dev['label'].value_counts()
porcentaje_dev = df_dev['label'].value_counts(normalize=True) * 100
for label in conteo_dev.index:
    print(f"   {label:12}: {conteo_dev[label]:5} muestras ({porcentaje_dev[label]:5.2f}%)")

# Test
print("\n📌 TEST:")
conteo_test = df_test['label'].value_counts()
porcentaje_test = df_test['label'].value_counts(normalize=True) * 100
for label in conteo_test.index:
    print(f"   {label:12}: {conteo_test[label]:5} muestras ({porcentaje_test[label]:5.2f}%)")


# ============================================
# 3. TABLA RESUMEN COMPARATIVA
# ============================================

print("\n" + "="*60)
print("TABLA RESUMEN - TODOS LOS SPLITS")
print("="*60)

# Crear DataFrame resumen
labels_unicos = ['true', 'false', 'mixture', 'unproven']
resumen_labels = pd.DataFrame({
    'Label': labels_unicos,
    'TRAIN': [conteo_train.get(label, 0) for label in labels_unicos],
    'DEV': [conteo_dev.get(label, 0) for label in labels_unicos],
    'TEST': [conteo_test.get(label, 0) for label in labels_unicos],
    'TOTAL': [
        conteo_train.get(label, 0) + conteo_dev.get(label, 0) + conteo_test.get(label, 0) 
        for label in labels_unicos
    ]
})

print("\n" + resumen_labels.to_string(index=False))

# Añadir fila de total
resumen_labels.loc['TOTAL'] = ['TOTAL', 
                                resumen_labels['TRAIN'].sum(), 
                                resumen_labels['DEV'].sum(), 
                                resumen_labels['TEST'].sum(),
                                resumen_labels['TOTAL'].sum()]
print("\n" + resumen_labels.to_string(index=False))