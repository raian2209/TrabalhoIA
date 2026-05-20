# Modelo treinado da MLP

Esta pasta guarda uma versao treinada da rede neural do trabalho e um exemplo de
inferencia em dados novos.

Arquivos:

- `salvar_modelo.py`: treina a MLP com os dados originais e salva os pesos.
- `modelo_mlp_frutas.json`: modelo treinado em formato JSON.
- `dados_teste.csv`: exemplos novos usados para inferencia.
- `inferir_modelo.py`: carrega o modelo salvo e classifica os dados de teste.
- `predicoes_teste.csv`: arquivo gerado com as predicoes dos dados de teste.

Para recriar o modelo:

```bash
cd TrabalhoRedesNeurais/modelo_treinado
python3 salvar_modelo.py
```

Para executar a inferencia:

```bash
cd TrabalhoRedesNeurais/modelo_treinado
python3 inferir_modelo.py
```
