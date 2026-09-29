# SMS Spam Detection

Projekat iz predmeta Veštačka inteligencija sa primenama.

**Student:** Bogdan Milojević, 568/2023

**Tema:** Transformer — SMS Spam Detection

Cilj je klasifikacija SMS poruka na `ham` (regularne poruke) i `spam` (neželjene poruke), pomoću sopstvenog malog Transformer modela implementiranog u PyTorch-u. Parametri modela biće nasumično inicijalizovani i trenirani na SMS podacima.

## Trenutno stanje

Postavljeni su okruženje, preuzimanje i učitavanje Kaggle skupa, ponovljiva obrada, stratifikovana podela podataka, tokenizacija, rečnik iz trening poruka, priprema nizova i PyTorch Dataset, kao i šest Jupyter sveski. Model i eksperimenti dolaze u narednim koracima. Trenutno nema istreniranog modela ni rezultata evaluacije.

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
python -c "import pandas, matplotlib, notebook, torch; print('Okruženje radi')"
python -m src.download_data
python -m notebook --notebook-dir=notebooks
```

Jupyter otvara lokalni interfejs u pregledaču. Sveske `01`–`03` prikazuju učitavanje, čišćenje i podelu podataka. `04_tokenization_vocabulary.ipynb` prikazuje tokene i ID-jeve, `05_sequences_and_padding.ipynb` dopunu i masku, a `06_pytorch_dataset.ipynb` jedan PyTorch primer i batch. Svaki notebook izvrši redom od prve ćelije. Server se zaustavlja u terminalu pritiskom na `Ctrl+C` i potvrdom ako je zatraži. Komanda `deactivate` izlazi iz virtuelnog okruženja.

## Biblioteke u ovoj fazi

- **pandas** — učitavanje i obrada tabelarnih podataka.
- **matplotlib** — crtanje grafikona.
- **notebook** — Jupyter interfejs za kombinovanje koda, objašnjenja i rezultata.
- **PyTorch (`torch`)** — tenzori, skup primera i batch-evi; kasnije i slojevi modela.

scikit-learn i MLflow dodaćemo kada implementiramo delove koji ih koriste. Biblioteka Hugging Face Transformers nije potrebna za planiranu arhitekturu: Transformer slojeve koristićemo iz PyTorch-a.

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

Rezultat je `['claim', 'your', 'free', 'prize', '!']`. Ovo je fiksno pravilo, bez učenja iz podataka. Rečnik iz narednog odeljka zavisi od trening poruka; tokom cross-validacije pravićemo ga ponovo za svaki trening fold.

## Rečnik i ID-jevi

`src/vocabulary.py` prima trening poruke, tokenizuje ih i svakom različitom tokenu dodeljuje ID. `<PAD>` ima ID `0`, a `<UNK>` ID `1`. Ostali tokeni se poređaju po opadajućoj učestalosti, pa abecedno kada imaju istu učestalost. ID je samo adresa tokena u rečniku, ne broj koji meri njegovo značenje.

Na malom primeru `Free prize now!` i `Free entry now.` dobijamo `free → 2`, `now → 3`, `! → 4`, `prize → 7`. Poruka `Free offer!` postaje `[2, 1, 4]`: `offer` nije viđen u primerima, pa dobija `<UNK>` ID `1`. Brojevi u ovom primeru nisu isti kao ID-jevi rečnika napravljenog iz celog trening skupa.

Sveska `04_tokenization_vocabulary.ipynb` prikazuje tabelu tih dodela i gradi pravi rečnik iz `data/processed/sms_train.csv`. Na 4.127 trening poruka on sadrži 7.842 unosa, uključujući dva posebna tokena. Izdvojeni test se ne čita pri građenju rečnika. `<PAD>` služi za dopunu nizova u sledećem odeljku.

## Priprema nizova i maska

`src/sequences.py` pretvara SMS u dve liste iste zadate dužine. Prva sadrži ID-jeve: ako poruka ima više od `max_length` tokena, zadržavamo prvih `max_length`; ako ima manje, dopunjavamo je ID-jem `<PAD> = 0`. Druga lista je `padding_mask`: `False` stoji uz stvarni token, a `True` uz dodatu nulu koju model kasnije treba da ignoriše. Nepoznati stvarni token ima ID `<UNK> = 1` i vrednost maske `False`.

Na malom rečniku iz prethodnog odeljka, `Free offer!` sa `max_length=6` postaje `[2, 1, 4, 0, 0, 0]`, a maska `[False, False, False, True, True, True]`. Početna podrazumevana dužina je 128. U 4.127 trening poruka medijana je 16 tokena, a samo četiri poruke imaju više od 128 tokena. To je provereno bez čitanja izdvojenog testa. Sveska `05_sequences_and_padding.ipynb` prikazuje i primer skraćivanja. Sledeći odeljak povezuje ove liste sa PyTorch-om.

## PyTorch primeri i batch-evi

`src/sms_dataset.py` pretvara jedan red tabele u tri tenzora: `input_ids` (`torch.long`), `padding_mask` (`torch.bool`) i `label` (`torch.long`). Klase imaju zasebno mapiranje `ham = 0`, `spam = 1`; to nisu ID-jevi reči. Dataset koristi prosleđeni rečnik, pa ćemo mu tokom cross-validacije dati rečnik odgovarajućeg trening fold-a.

PyTorch `DataLoader` spaja više primera u batch. Za dve poruke i dužinu 8, `input_ids` i `padding_mask` imaju oblik `(2, 8)`, a `label` oblik `(2,)`. Prva dimenzija je broj poruka, druga broj pozicija u svakoj poruci. `06_pytorch_dataset.ipynb` prikazuje taj mali primer i prvi batch iz trening skupa. Izdvojeni test se ne koristi za izgradnju rečnika. Transformer i treniranje još nisu implementirani.
