"""import pandas as pd

# Los nombres de las columnas según el script de Hugging Face
columnas = [
    'claim_id',          # ID único de la afirmación
    'claim',             # La afirmación/noticia a verificar
    'date_published',    # Fecha de publicación
    'explanation',       # Explicación del fact-checker (¡útil para entrenar!)
    'fact_checkers',     # Quiénes hicieron el fact-check
    'main_text',         # Texto completo del artículo
    'sources',           # Fuentes/URLs de evidencia
    'label',             # Etiqueta: true, false, mixture, unproven
    'subjects'           # Temas de salud relacionados
]"""

import pandas as pd

columnas = [
    'claim_id', 'claim', 'date_published', 'explanation',
    'fact_checkers', 'main_text', 'sources', 'label', 'subjects'
]

df_train = pd.read_csv('/hhome/nlp208/PUBHEALTH/PUBHEALTH/dev.tsv', 
                       sep='\t', 
                       names=columnas, 
                       skiprows=1)

df_reduced = df_train[['claim_id', 'claim', 'explanation', 'main_text', 'label']]

df_reduced.to_csv('/hhome/nlp208/PUBHEALTH/PUBHEALTH/dev_reduced.csv', 
                   index=False, 
                   encoding='utf-8')

print("✅ Guardado: dev_reduced.csv")