# SMS Spam Detection

Projekat iz predmeta Veštačka inteligencija sa primenama.

**Student:** Bogdan Milojević, 568/2023

**Tema:** Transformer — SMS Spam Detection

Cilj je klasifikacija SMS poruka na `ham` (regularne poruke) i `spam` (neželjene poruke), pomoću sopstvenog malog Transformer modela implementiranog u PyTorch-u. Parametri modela biće nasumično inicijalizovani i trenirani na SMS podacima.

## Trenutno stanje

Postavljeni su okruženje, preuzimanje i učitavanje Kaggle skupa, ponovljiva obrada, stratifikovana podela podataka i osnovna tokenizacija, kao i tri Jupyter sveske. Rečnik, model i eksperimenti dolaze u narednim koracima. Trenutno nema istreniranog modela ni rezultata evaluacije.

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
python -m src.download_data
python -m notebook --notebook-dir=notebooks
```

Jupyter otvara lokalni interfejs u pregledaču. `01_data_loading.ipynb` prikazuje originalni CSV, a `02_eda_preprocessing.ipynb` proverava nedostajuće vrednosti, čišćenje, raspodelu klasa i dužine poruka, a `03_train_test_split.ipynb` prikazuje podelu. Svaki notebook izvrši redom od prve ćelije. Server se zaustavlja u terminalu pritiskom na `Ctrl+C` i potvrdom ako je zatraži. Komanda `deactivate` izlazi iz virtuelnog okruženja.

## Biblioteke u ovoj fazi

- **pandas** — učitavanje i obrada tabelarnih podataka.
- **matplotlib** — crtanje grafikona.
- **notebook** — Jupyter interfejs za kombinovanje koda, objašnjenja i rezultata.

PyTorch, scikit-learn i MLflow dodaćemo kada implementiramo delove koji ih koriste. Biblioteka Hugging Face Transformers nije potrebna za planiranu arhitekturu: Transformer slojeve koristićemo iz PyTorch-a.

## Dataset

Planirani izvor: [SMS Spam Collection Dataset na Kaggle-u](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Skripta `python -m src.download_data` preuzima originalnu ZIP arhivu u `data/raw/`. Fajl je izuzet iz Git-a. Očekivani SHA-256 arhive je `3e05b8e6e1e8fc9aef3ca69399a1bf3849a22084c8401d5d5d592e6c9a0e422b`. Ako se izvor promeni, skripta prijavljuje razliku i ne zamenjuje postojeći fajl.

Učitani CSV ima 5.572 reda: 4.825 `ham` i 747 `spam`. Pored `v1` i `v2` postoje tri dodatne kolone. One su uglavnom prazne, ali u 50 redova sadrže nastavke poruka. `src/preprocessing.py` spaja delove zarezom, uklanja samo spoljne praznine i proverava da ista poruka nema suprotne oznake. Zatim uklanja 414 potpuno identičnih poruka. Rezultat ima 5.158 redova: 4.516 `ham` i 642 `spam`. Interpunkcija i velika slova se čuvaju. Duplikati se uklanjaju pre buduće podele na trening i test da se ista poruka ne pojavi na obe strane.

Obrađeni CSV može se ponovo napraviti komandom:

```bash
python -m src.preprocessing
```

Fajl `data/processed/sms_clean.csv` nastaje lokalno i izuzet je iz Git-a.

## Podela na trening i test

Komanda `python -m src.split_data` ponavlja čišćenje originalnog skupa i pravi `data/processed/sms_train.csv` i `data/processed/sms_test.csv`. Obe datoteke su lokalne i izuzete iz Git-a. Podela je stratifikovana po klasi: nasumično se bira približno 20% iz svake klase, uz fiksno seme 42. Trening ima 4.127 poruka (3.613 `ham`, 514 `spam`), a izdvojeni test 1.031 poruku (903 `ham`, 128 `spam`). Nema istog teksta u oba dela.

Test čuvamo za završnu procenu modela. Rečnik tokena i sve parametre koji se uče iz podataka pravićemo samo iz trening dela; tokom cross-validacije iz odgovarajućeg trening fold-a. Model još nije implementiran.

## Tokenizacija

`src/tokenization.py` pretvara jednu SMS poruku u listu tokena. Tekst se pretvara u mala slova, nizovi slova ili cifara ostaju zajedno, a svaki znak interpunkcije postaje zaseban token. Originalni tekst u CSV-u se ne menja.

```bash
python -c "from src.tokenization import tokenize_sms; print(tokenize_sms('Claim your free prize!'))"
```

Rezultat je `['claim', 'your', 'free', 'prize', '!']`. Ovo je fiksno pravilo, bez učenja iz podataka. Sledeći korak je da napravimo rečnik samo iz trening poruka i pretvorimo tokene u ID-jeve. U cross-validaciji rečnik ćemo ponovo graditi za svaki trening fold.
