import pandas as pd

df = pd.read_csv("stopwords.csv")
old_list = df.data.tolist()
new_list = ['accion', 'actividad', 'apoyar', 'apoyo', 'beneficio', 'capacidad', 'competencia', 'control', 'correo', 'crecimiento', 'diferente',
            'diferentes', 'ejecucion', 'elaboracion', 'encontrar', 'establecido', 'forma', 'funcion', 'general', 'generar', 'habilidad',
            'hacer', 'mejora', 'mejorar', 'objetivo', 'office', 'oportunidad', 'pais', 'paquete', 'permitir', 'pertenecer', 'presentacion',
            'querer', 'relacion', 'relacionado', 'resultado']

print(len(old_list), len(new_list))
old_list += new_list
print(len(old_list), len(new_list))
df = pd.DataFrame(data=old_list, columns=["data"])
print(df.tail())
df.to_csv("stopwords.csv")

if __name__ == "__main__":
    exit(0)
