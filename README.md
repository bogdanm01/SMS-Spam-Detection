# SMS Spam Detection

Streamlit aplikacija klasifikuje SMS poruke kao `ham` (regularne) ili `spam` (neželjene). Obučeni model je uključen u repozitorijum, pa treniranje nije potrebno za pokretanje aplikacije.

## Pokretanje

Potreban je Docker sa podrškom za Compose. Pokreni komande iz korena repozitorijuma:

```bash
docker compose up --build
```

Otvori [http://127.0.0.1:8501](http://127.0.0.1:8501), unesi SMS poruku i klikni **Proveri poruku**.

Za zaustavljanje pritisni `Ctrl+C`, a zatim ukloni kontejner:

```bash
docker compose down
```

## Skup podataka

Korišćen je [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), sa oznakama `ham` i `spam`. Skripta `python -m src.download_data` preuzima originalnu arhivu u `data/raw/`. Podaci nisu uključeni u repozitorijum i nisu potrebni za pokretanje aplikacije sa već obučenim modelom.

## Tehnologije

Python 3.12, PyTorch (sopstveni Transformer model), pandas, Matplotlib, Jupyter Notebook, MLflow, Streamlit i Docker Compose.

## Struktura projekta

```text
SMS-Spam-Detection/
├── app.py                 # Streamlit aplikacija
├── src/                   # Obrada podataka, model, treniranje i predviđanje
├── models/
│   └── final_model.pt     # Obučeni model koji koristi aplikacija
├── notebooks/             # Jupyter sveske za analizu i objašnjenje procesa
├── data/
│   ├── raw/               # Originalni podaci
│   └── processed/         # Obrađeni podaci
├── results/               # Metrike, poređenje i izbor modela
├── .streamlit/            # Podešavanja interfejsa
├── Dockerfile             # Docker slika aplikacije
├── compose.yaml           # Pokretanje kontejnera
├── requirements-app.txt   # Zavisnosti aplikacije
├── requirements.txt       # Zavisnosti za razvoj i treniranje
├── requirements-lock.txt  # Zaključane verzije zavisnosti
└── README.md
```
