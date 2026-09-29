# SMS Spam Detection

Projekat iz predmeta Veštačka inteligencija sa primenama.

**Student:** Bogdan Milojević, 568/2023

**Tema:** Transformer — SMS Spam Detection

Cilj je klasifikacija SMS poruka na `ham` (regularne poruke) i `spam` (neželjene poruke), pomoću sopstvenog malog Transformer modela implementiranog u PyTorch-u. Parametri modela biće nasumično inicijalizovani i trenirani na SMS podacima.

## Trenutno stanje

Projekat je u početnoj fazi: postavljeni su struktura direktorijuma i Python okruženje za analizu podataka. Dataset još nije preuzet. Notebook, preprocessing, model i eksperimenti biće dodavani u narednim koracima. Trenutno nema istreniranog modela ni rezultata evaluacije.

## Planirana arhitektura

```text
SMS → tokenizacija → identifikatori tokena
    → embedding + pozicije → Transformer encoder
    → prosek reprezentacija stvarnih tokena → klasifikacioni sloj
```

Plan obuhvata rečnik napravljen isključivo iz trening dela, posebne `<PAD>` i `<UNK>` tokene, padding masku i treniranje embedding-a, Transformer blokova i klasifikacionog sloja zajedno. Koristićemo PyTorch komponente za attention i encoder blokove; ne preuzimamo prethodno obučene težine.

Početna konfiguracija za prvi probni trening: dimenzija reprezentacije 64, dva encoder bloka, četiri attention glave, feed-forward dimenzija 128, dropout 0,1 i maksimalna dužina 128 tokena. To je početna postavka za proveru, a ne izabrani najbolji model.

Planirano je poređenje najmanje pet konfiguracija uz cross-validaciju i MLflow evidenciju. Rečnik se gradi zasebno na trening delu svakog fold-a, a model se ponovo inicijalizuje za svaki trening. Konačne konfiguracije definisaćemo nakon provere početnog modela. Izdvojeni test skup ne koristimo za njihov izbor.

## Struktura

```text
SMS-Spam-Detection/
├── data/
│   ├── raw/                 # Originalni podaci
│   └── processed/           # Podaci dobijeni obradom
├── notebooks/              # Jupyter sveske za analizu
├── src/                    # Python moduli za obradu i model
├── results/                # Budući rezultati i grafikoni
├── .gitignore              # Šta Git ne treba da prati
├── .python-version         # Python 3.12
├── requirements.txt        # Direktne zavisnosti
├── requirements-lock.txt   # Tačne verzije instaliranog okruženja
└── README.md
```

Prazni direktorijumi sadrže `.gitkeep` fajl da bi njihova struktura mogla da se sačuva u Git-u. Taj fajl nema ulogu u Python programu.

## Lokalno okruženje

Koristi se Python 3.12. Komande se izvršavaju iz korena projekta.

### macOS i Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Za rekonstrukciju tačnih verzija svih zavisnosti iz proverenog okruženja, umesto poslednje komande koristiti:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` navodi biblioteke koje neposredno koristimo, a `requirements-lock.txt` beleži i njihove pomoćne zavisnosti. Lock fajl predstavlja provereno lokalno okruženje; druge platforme proveravaćemo kada budu potrebne.

## Provera i Jupyter

U aktiviranom okruženju:

```bash
python -c "import pandas, matplotlib, notebook; print('Okruženje radi')"
python -m notebook --notebook-dir=notebooks
```

Jupyter otvara lokalni interfejs u pregledaču. Notebook za naš dataset dodajemo u narednom koraku. Server se zaustavlja u terminalu pritiskom na `Ctrl+C` i potvrdom ako je zatraži. Komanda `deactivate` izlazi iz virtuelnog okruženja.

## Biblioteke u ovoj fazi

- **pandas** — učitavanje i obrada tabelarnih podataka.
- **matplotlib** — crtanje grafikona.
- **notebook** — Jupyter interfejs za kombinovanje koda, objašnjenja i rezultata.

PyTorch, scikit-learn i MLflow dodaćemo kada implementiramo delove koji ih koriste. Biblioteka Hugging Face Transformers nije potrebna za planiranu arhitekturu: Transformer slojeve koristićemo iz PyTorch-a.

## Dataset

Planirani izvor: [SMS Spam Collection Dataset na Kaggle-u](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Preuzimanje, provera i analiza podataka predmet su sledećih koraka. Statistike ćemo navesti nakon pregleda stvarnog fajla.
