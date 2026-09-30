# SMS Spam Detection

Projekat iz predmeta Veštačka inteligencija sa primenama.

**Student:** Bogdan Milojević, 568/2023

**Tema:** Transformer — SMS Spam Detection

Cilj je klasifikacija SMS poruka na `ham` (regularne poruke) i `spam` (neželjene poruke), pomoću sopstvenog malog Transformer modela implementiranog u PyTorch-u. Model se inicijalizuje nasumično i trenira na SMS podacima.

## Trenutno stanje

Postavljeni su okruženje, preuzimanje i učitavanje Kaggle skupa, ponovljiva obrada, stratifikovana podela podataka, tokenizacija, rečnik iz trening poruka, priprema nizova i PyTorch Dataset, kao i tri objedinjene Jupyter sveske. Prvi prolaz kroz sopstveni Transformer model i jedan probni trening korak na dve izmišljene poruke su implementirani. Petlja za treniranje nad celim trening skupom je implementirana i prikazana kroz dve probne epohe. Treća sveska prikazuje validaciju, 5-fold cross-validaciju, poređenje pet arhitektura i izbor konfiguracije. Nova instanca izabranog `wide` modela trenirana je dve epohe na celom trening skupu, sačuvana lokalno i potom proverena na izdvojenom test skupu. Nova SMS poruka može se proveriti iz terminala ili kroz Streamlit interfejs.

## Arhitektura modela

```text
SMS → tokenizacija → identifikatori tokena
    → embedding + pozicije → Transformer encoder
    → prosek reprezentacija stvarnih tokena → klasifikacioni sloj
```

Arhitektura koristi rečnik napravljen isključivo iz trening dela, posebne `<PAD>` i `<UNK>` tokene i padding masku. Embedding, Transformer blokovi i klasifikacioni sloj treniraju se zajedno. Encoder blokovi su PyTorch komponente; ne preuzimamo prethodno obučene težine.

Podrazumevana konfiguracija modela za prvi probni trening: dimenzija reprezentacije 64, dva encoder bloka, četiri attention glave, feed-forward dimenzija 128, dropout 0,1 i maksimalna dužina 128 tokena. To je početna postavka za proveru, a ne izabrani najbolji model.

Pet konfiguracija poredimo pomoću istih pet fold-ova i beležimo eksperimente u MLflow-u. Rečnik se gradi zasebno na trening delu svakog fold-a, a model se ponovo inicijalizuje za svaki trening. Konfiguracije menjaju širinu ili dubinu mreže uz iste ostale uslove treniranja. Izdvojeni test skup ne koristimo za njihov izbor.

## Struktura

```text
SMS-Spam-Detection/
├── data/
│   ├── raw/                 # Originalni podaci
│   └── processed/           # Podaci dobijeni obradom
├── notebooks/              # Jupyter sveske za analizu
├── src/                    # Python moduli za obradu i model
├── results/                # Tabele rezultata i odluka o modelu
├── models/                 # Završni checkpoint za predviđanje
├── app.py                  # Streamlit interfejs
├── .streamlit/             # Boje interfejsa
├── Dockerfile              # Slika za pokretanje aplikacije
├── compose.yaml            # Lokalni kontejner i port
├── requirements-app.txt    # Web zavisnosti aplikacije
├── .dockerignore           # Fajlovi van Docker build konteksta
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

Jupyter otvara lokalni interfejs u pregledaču. Tri sveske prate tok projekta:

1. `01_data_analysis.ipynb` — učitavanje, čišćenje, EDA grafikoni i podela na trening i izdvojeni test.
2. `02_text_preparation.ipynb` — tokenizacija, rečnik iz trening podataka, česti tokeni, dopuna i dužine nizova.
3. `03_dataset_and_model.ipynb` — PyTorch Dataset, model, probni trening, validacija, poređenje i izbor konfiguracije.

Svaku svesku izvrši redom od prve ćelije. Pre druge sveske pokreni `python -m src.split_data` da nastane lokalni `sms_train.csv`; prva sveska podelu prikazuje u memoriji i ne upisuje fajlove. Server se zaustavlja u terminalu pritiskom na `Ctrl+C` i potvrdom ako je zatraži. Komanda `deactivate` izlazi iz virtuelnog okruženja.

## Biblioteke u ovoj fazi

- **pandas** — učitavanje i obrada tabelarnih podataka.
- **matplotlib** — crtanje grafikona.
- **notebook** — Jupyter interfejs za kombinovanje koda, objašnjenja i rezultata.
- **PyTorch (`torch`)** — tenzori, batch-evi, Transformer model i petlja za trening epohe.
- **MLflow** — lokalna evidencija parametara i metrika za poređenje konfiguracija.
- **Streamlit** — lokalni korisnički interfejs za proveru novih SMS poruka.

Za podelu podataka i metrike koristimo postojeći kod; scikit-learn ne pozivamo direktno. MLflow ga instalira kao svoju zavisnost. Biblioteka Hugging Face Transformers nije potrebna za našu arhitekturu: Transformer slojeve koristimo iz PyTorch-a.

## Dataset

Planirani izvor: [SMS Spam Collection Dataset na Kaggle-u](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

Skripta `python -m src.download_data` preuzima originalnu ZIP arhivu u `data/raw/`. Fajl je izuzet iz Git-a. Očekivani SHA-256 arhive je `3e05b8e6e1e8fc9aef3ca69399a1bf3849a22084c8401d5d5d592e6c9a0e422b`. Ako se izvor promeni, skripta prijavljuje razliku i ne zamenjuje postojeći fajl.

Učitani CSV ima 5.572 reda: 4.825 `ham` i 747 `spam`. Pored `v1` i `v2` postoje tri dodatne kolone. One su uglavnom prazne, ali u 50 redova sadrže nastavke poruka. `src/preprocessing.py` spaja delove zarezom, uklanja samo spoljne praznine i proverava da ista poruka nema suprotne oznake. Zatim uklanja 414 potpuno identičnih poruka. Rezultat ima 5.158 redova: 4.516 `ham` i 642 `spam`. Interpunkcija i velika slova se čuvaju. Duplikati se uklanjaju pre buduće podele na trening i test da se ista poruka ne pojavi na obe strane. Notebook `01_data_analysis.ipynb` prikazuje i uklonjene duplikate po klasi (309 `ham`, 105 `spam`) i raspodelu dužina normalizovanu unutar svake klase. Ovi prikazi opisuju čišćenje celog skupa; za odluke o tokenima i dužini modela koristimo samo trening deo.

Obrađeni CSV može se ponovo napraviti komandom:

```bash
python -m src.preprocessing
```

Fajl `data/processed/sms_clean.csv` nastaje lokalno i izuzet je iz Git-a.

## Podela na trening i test

Komanda `python -m src.split_data` ponavlja čišćenje originalnog skupa i pravi `data/processed/sms_train.csv` i `data/processed/sms_test.csv`. Obe datoteke su lokalne i izuzete iz Git-a. Podela je stratifikovana po klasi: nasumično se bira približno 20% iz svake klase, uz fiksno seme 42. Trening ima 4.127 poruka (3.613 `ham`, 514 `spam`), a izdvojeni test 1.031 poruku (903 `ham`, 128 `spam`). Nema istog teksta u oba dela.

Test čuvamo za završnu procenu modela. Rečnik tokena i sve parametre koji se uče iz podataka pravimo samo iz trening dela; tokom cross-validacije iz odgovarajućeg trening fold-a. Posle izbora konfiguracije novi model je treniran na celom trening delu, bez otvaranja testa.

## Tokenizacija

`src/tokenization.py` pretvara jednu SMS poruku u listu tokena. Tekst se pretvara u mala slova, nizovi slova ili cifara ostaju zajedno, a svaki znak interpunkcije postaje zaseban token. Originalni tekst u CSV-u se ne menja.

```bash
python -c "from src.tokenization import tokenize_sms; print(tokenize_sms('Claim your free prize!'))"
```

Rezultat je `['claim', 'your', 'free', 'prize', '!']`. Ovo je fiksno pravilo, bez učenja iz podataka. Rečnik iz narednog odeljka zavisi od trening poruka; tokom cross-validacije pravićemo ga ponovo za svaki trening fold.

## Rečnik i ID-jevi

`src/vocabulary.py` prima trening poruke, tokenizuje ih i svakom različitom tokenu dodeljuje ID. `<PAD>` ima ID `0`, a `<UNK>` ID `1`. Ostali tokeni se poređaju po opadajućoj učestalosti, pa abecedno kada imaju istu učestalost. ID je samo adresa tokena u rečniku, ne broj koji meri njegovo značenje.

Na malom primeru `Free prize now!` i `Free entry now.` dobijamo `free → 2`, `now → 3`, `! → 4`, `prize → 7`. Poruka `Free offer!` postaje `[2, 1, 4]`: `offer` nije viđen u primerima, pa dobija `<UNK>` ID `1`. Brojevi u ovom primeru nisu isti kao ID-jevi rečnika napravljenog iz celog trening skupa.

Sveska `02_text_preparation.ipynb` prikazuje tabelu tih dodela i gradi pravi rečnik iz `data/processed/sms_train.csv`. Na 4.127 trening poruka on sadrži 7.842 unosa, uključujući dva posebna tokena. Izdvojeni test se ne čita pri građenju rečnika. Ista sveska prikazuje najčešće tokene po klasi kao udeo trening poruka koje sadrže token. `<PAD>` služi za dopunu nizova u sledećem odeljku.

## Priprema nizova i maska

`src/sequences.py` pretvara SMS u dve liste iste zadate dužine. Prva sadrži ID-jeve: ako poruka ima više od `max_length` tokena, zadržavamo prvih `max_length`; ako ima manje, dopunjavamo je ID-jem `<PAD> = 0`. Druga lista je `padding_mask`: `False` stoji uz stvarni token, a `True` uz dodatu nulu koju model kasnije treba da ignoriše. Nepoznati stvarni token ima ID `<UNK> = 1` i vrednost maske `False`.

Na malom rečniku iz prethodnog odeljka, `Free offer!` sa `max_length=6` postaje `[2, 1, 4, 0, 0, 0]`, a maska `[False, False, False, True, True, True]`. Početna podrazumevana dužina je 128. U 4.127 trening poruka medijana je 16 tokena, a samo četiri poruke imaju više od 128 tokena. To je provereno bez čitanja izdvojenog testa. Sveska `02_text_preparation.ipynb` prikazuje primer skraćivanja i histogram dužina prema našem tokenizer-u, sa granicom od 128 tokena. Sledeći odeljak povezuje ove liste sa PyTorch-om.

## PyTorch primeri i batch-evi

`src/sms_dataset.py` pretvara jedan red tabele u tri tenzora: `input_ids` (`torch.long`), `padding_mask` (`torch.bool`) i `label` (`torch.long`). Klase imaju zasebno mapiranje `ham = 0`, `spam = 1`; to nisu ID-jevi reči. Dataset koristi prosleđeni rečnik, pa ćemo mu tokom cross-validacije dati rečnik odgovarajućeg trening fold-a.

PyTorch `DataLoader` spaja više primera u batch. Za dve poruke i dužinu 8, `input_ids` i `padding_mask` imaju oblik `(2, 8)`, a `label` oblik `(2,)`. Prva dimenzija je broj poruka, druga broj pozicija u svakoj poruci. `03_dataset_and_model.ipynb` prikazuje taj mali primer i prvi batch iz trening skupa. Izdvojeni test se ne koristi za izgradnju rečnika. Transformer model je implementiran; u svesci sledi jedan probni trening korak.

## Prvi prolaz kroz model

`src/model.py` definiše `SMSClassifier`. ID-jevi tokena postaju vektori dimenzije 64, dodaje im se naučivi vektor pozicije, a dva Transformer encoder bloka računaju reprezentacije tokena u kontekstu. Maska označava dopunske pozicije; one se ne koriste kao ključevi u attention-u niti ulaze u prosek reprezentacija cele poruke. Linearni sloj iz tog proseka vraća po dva sirova skora (`logits`), redom za `ham = 0` i `spam = 1`. Za batch od 2 poruke izlaz ima oblik `(2, 2)`.

Sveska `03_dataset_and_model.ipynb` prikazuje te oblike i jedan izlaz sa tek inicijalizovanim parametrima. Skorovi i vrednosti dobijene funkcijom `softmax` još nisu pouzdana predviđanja. U ovoj celini se ne računa funkcija gubitka, ne menjaju se parametri i ne koristi se izdvojeni test skup.

## Jedan trening korak

`src/training.py` povezuje `SMSClassifier`, tačne oznake i optimizer za jedan batch. `CrossEntropyLoss` direktno prima dva sirova skora po poruci i celobrojnu oznaku `ham = 0` ili `spam = 1`. Posle računanja gubitka, `backward()` računa gradijente, a `AdamW.step()` jednom ažurira parametre. Stari gradijenti se brišu pre tog prolaza.

Sveska `03_dataset_and_model.ipynb` prikazuje ovu promenu na dve izmišljene poruke i proverava da su se težine klasifikatora promenile. Jedan korak ne daje pouzdano naučen model niti metriku kvaliteta. Posle tog prikaza sveska prolazi kroz sve trening batch-eve tokom epoha, uz odvojenu validaciju.

## Trening epoha

`train_one_epoch` iz `src/training.py` poziva `train_one_batch` za svaki batch iz `DataLoader`-a i vraća prosečan gubitak po poruci. Pošto poslednji batch može biti manji, svaki batch gubitak množi brojem njegovih poruka pre deljenja ukupnim brojem primera.

Treća sveska koristi svih 4.127 trening poruka, `batch_size=32` i `shuffle=True`: jedna epoha ima 129 batch-eva, odnosno 129 ažuriranja parametara. Prikazane su dve demonstracione epohe sa novo inicijalizovanim modelom. Gubitak nad trening porukama nije rezultat evaluacije. Zatim sveska proverava model na izdvojenom delu trening skupa; završni test ostaje netaknut.

## Prva validacija

Sveska `03_dataset_and_model.ipynb` poziva postojeći `split_sms` na `sms_train.csv` i dobija deo za učenje i validaciju, stratifikovano po klasama. Novi rečnik i sveže inicijalizovani model koriste samo deo za učenje. Validacioni podaci ne ulaze u rečnik niti menjaju parametre. `sms_test.csv` se ne čita.

`src/evaluation.py` računa validacioni gubitak, tačnost, preciznost, odziv i F1 za `spam`, kao i TN/FP/FN/TP za matricu zabune. Tokom provere model je u evaluacionom režimu i gradijenti se ne računaju. Sveska prikazuje rezultate posle tri demonstracione epohe i poredi ih sa naivnim pravilom „sve je ham“. Ovo je samo jedna podela; naredni odeljak uvodi cross-validaciju za pouzdanije poređenje.

## Prva cross-validacija

`src/cross_validation.py` deli samo `sms_train.csv` na pet stratifikovanih fold-ova. Svaka trening poruka ulazi tačno jednom u validacioni fold. Za svaki fold rečnik se pravi iz preostala četiri dela, a model i optimizer se iznova inicijalizuju. Kroz tri demonstracione epohe beležimo trening gubitak i validacione metrike posle svake epohe. Ovo je pet zasebnih treniranja iste početne konfiguracije.

Treća sveska prikazuje broj `ham` i `spam` poruka po foldu, pojedinačne rezultate i prosek sa standardnom devijacijom kroz fold-ove. U ovom pokretanju F1 za spam posle treće epohe iznosi 0,822–0,899 po foldovima, prosečno 0,862 (standardna devijacija 0,035). To još nije poređenje konfiguracija niti završna ocena. Modeli iz fold-ova se ne čuvaju kao konačni model, a `sms_test.csv` se ne učitava.

## Poređenje konfiguracija

`python -m src.experiments` čita samo `data/processed/sms_train.csv`, pokreće pet konfiguracija kroz istih pet stratifikovanih fold-ova i čuva po jedan red za svaki fold i epohu u `results/config_comparison.csv`. Jedna konfiguracija znači pet nezavisnih treniranja, pa ceo eksperiment sadrži 25 treniranja. Svako koristi tri epohe, batch veličine 32, stopu učenja 0,001, maksimalnu dužinu 128 i dropout 0,1. Rečnik, model i optimizer nastaju iznova unutar svakog fold-a.

| Konfiguracija | Dimenzija vektora | Encoder slojevi | Attention glave | Feed-forward dimenzija |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 64 | 2 | 4 | 128 |
| `narrow` | 32 | 2 | 4 | 64 |
| `wide` | 96 | 2 | 4 | 192 |
| `shallow` | 64 | 1 | 4 | 128 |
| `deep` | 64 | 3 | 4 | 128 |

Treća sveska čita sačuvani CSV i prikazuje F1 za spam po foldu, prosek i standardnu devijaciju za svaku konfiguraciju, zajedno sa odzivom i prosečnim brojem parametara. Poređenje unapred koristi F1 posle treće epohe kao glavnu metriku. Završni test skup ne učestvuje u izboru. Konačno treniranje jednog izabranog modela sledi kasnije.

## MLflow evidencija

Komanda `python -m src.experiments` ponovo trenira svih 25 modela. Tokom tog pokretanja otvara po jedan MLflow run za svaku konfiguraciju u eksperimentu `sms-spam-transformer`. Beleži arhitekturne i trening parametre, metrike svakog fold-a po epohi, prosečne metrike i standardnu devijaciju F1 posle treće epohe. U svakom run-u čuva i `fold_history.csv`. Ova evidencija nastaje tokom treniranja; raniji CSV nije retroaktivno unet kao da predstavlja novo treniranje.

Podrazumevano se MLflow metapodaci čuvaju u lokalnom `mlflow.db`, a artefakti u `mlruns/`. Obe lokacije su izuzete iz Git-a. Ako je postavljen `MLFLOW_TRACKING_URI`, skripta koristi tu adresu umesto lokalne baze. Za pregled lokalnih rezultata, iz korena projekta pokreni:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050
```

Zatim otvori `http://127.0.0.1:5050` i izaberi eksperiment `sms-spam-transformer`. Svaki red je jedna konfiguracija; kolona `mean_f1` daje prosek pet fold-ova posle treće epohe, `std_f1` njihovo rasipanje, a metrike `fold_1_f1` do `fold_5_f1` prikazuju tok kroz epohe. MLflow run nije završni sačuvani model.

## Izbor konfiguracije

`python -m src.select_model` čita `results/config_comparison.csv` i zapisuje odluku u `results/selected_model.json`. Prvo bira arhitekturu sa najvećim prosečnim F1 za spam posle **unapred određene treće epohe**. To je `wide` sa F1 **0,894**. Zatim među epohama samo te arhitekture bira onu sa najvećim prosečnim F1: **drugu**, sa F1 **0,901** i standardnom devijacijom **0,016**. U slučaju jednakog rezultata bira raniju epohu.

Druga epoha ima prosečan precision 0,965 i recall 0,846; treća ima precision 0,931 i recall 0,866. Dakle, ovaj izbor daje bolji F1, ali propušta nešto više spam poruka. `shallow` je jednostavnija alternativa, ali ima niži prosečan F1 posle treće epohe (0,882). Razlike su male i ovi brojevi su validacione procene, ne garantovan rezultat na novim porukama. Odluka se ne menja na osnovu izdvojenog testa.

Posle izbora napravili smo novu instancu `wide` modela i nov rečnik iz svih 4.127 trening poruka. Nijedan model iz cross-validacije nije postao konačan model.

## Konačno treniranje i čuvanje modela

`python -m src.train_final` čita samo `sms_train.csv` i `results/selected_model.json`, pravi nov model sa nasumično inicijalizovanim težinama i trenira ga dve epohe. Koristi isti batch od 32, stopu učenja 0,001, maksimalnu dužinu 128 i seme 42 kao eksperiment za izbor. Gubitak treniranja po epohi upisuje u `results/final_training.csv` i u zaseban MLflow eksperiment `sms-spam-final-training`.

Konačni rečnik ima **7.842 tokena**, a model **914.882 parametra**. Više parametara nego u pojedinačnim CV foldovima potiče od većeg rečnika iz celog trening skupa. Trening gubitak se smanjio sa **0,1857** na **0,0657**; to pokazuje da model uči trening poruke, ali nije mera rada na neviđenim porukama.

`models/final_model.pt` sadrži `state_dict` naučenih težina, rečnik tokena, arhitekturu, maksimalnu dužinu, mapiranje `ham=0`/`spam=1` i podatke o treniranju. `src/model_io.py` iz tih podataka ponovo pravi `SMSClassifier`, učitava težine uz `weights_only=True` i prebacuje ga u režim za predviđanje. Skripta proverava da učitani model daje isti izlaz kao model neposredno posle treniranja. Završni checkpoint više nije ignorisan i treba ga dodati u Git uz Docker fajlove; možeš ga ponovo napraviti pokretanjem gornje komande nakon pripreme podataka.

## Završni test

`python -m src.evaluate_final` učitava sačuvani `models/final_model.pt` i `sms_test.csv`. Proverava da se tekstovi treninga i testa ne preklapaju, koristi rečnik i maksimalnu dužinu iz checkpoint-a, ne menja težine i upisuje `results/test_metrics.json`.

Na **1.031** izdvojenoj poruci rezultat je: accuracy **0,9767**, precision za spam **0,9815**, recall **0,8281** i F1 **0,8983**. Matrica zabune ima **901 TN**, **2 FP**, **22 FN** i **106 TP**. Dakle, model je propustio 22 od 128 spam poruka, uz samo dve regularne poruke pogrešno označene kao spam. Treća sveska prikazuje ove metrike i matricu. Test rezultat ne koristimo da ponovo biramo arhitekturu ili broj epoha.

## Predviđanje za novu poruku

Kada je lokalni checkpoint već napravljen, iz korena projekta pokreni:

```bash
python -m src.predict "Claim your free prize now!"
```

`predict_sms` koristi rečnik, maksimalnu dužinu i mapiranje klasa iz checkpoint-a. Poruka prolazi kroz istu tokenizaciju i dopunu kao poruke tokom treniranja; model je u režimu evaluacije i njegovi parametri se ne menjaju. Funkcija vraća oznaku `ham` ili `spam` i dva softmax skora. Skorovi sabrani daju 1, ali ih ne treba tumačiti kao provereno kalibrisanu verovatnoću.

## Streamlit aplikacija

Iz korena projekta, u aktiviranom okruženju i nakon što postoji `models/final_model.pt`, pokreni:

```bash
python -m streamlit run app.py
```

Streamlit će prikazati lokalnu adresu, obično `http://localhost:8501`. U polje unesi SMS poruku i klikni **Proveri poruku**. Aplikacija poziva istu funkciju `predict_sms` kao terminalska komanda, kešira učitani model između ponovnih izvršavanja i prikazuje klasu i oba skora. Ako checkpoint nedostaje, prikazuje komandu `python -m src.train_final`. Server se zaustavlja sa `Ctrl+C` u terminalu.

## Docker

Docker pokreće samo aplikaciju za predviđanje; treniranje, Jupyter i MLflow ostaju u lokalnom Python okruženju. Dockerfile instalira CPU izdanje PyTorch-a i web zavisnosti iz `requirements-app.txt`. U sliku se kopira `models/final_model.pt`: završni checkpoint sadrži težine modela, rečnik i konfiguraciju potrebnu za predviđanje. Dodaj taj fajl u naredni Git commit zajedno sa Docker fajlovima; posle slanja na GitHub aplikacija će moći da se izgradi iz sveže kloniranog repozitorijuma bez ponovnog treniranja. Ostali checkpoint-i i lokalni podaci ostaju izuzeti.

Iz korena projekta pokreni:

```bash
docker compose up --build
```

Otvori `http://127.0.0.1:8501`. Port je dostupan samo sa ovog računara. Kontejner koristi checkpoint već upakovan u sliku; ne zavisi od lokalnog `models/` direktorijuma tokom pokretanja. Ako zameniš završni checkpoint novim treniranjem, ponovo izgradi sliku sa `docker compose up --build`.

Ako je port 8501 zauzet lokalno pokrenutim Streamlit-om, izaberi drugi port, na primer `SMS_APP_PORT=8502 docker compose up --build`, pa otvori `http://127.0.0.1:8502`.

Za zaustavljanje pritisni `Ctrl+C`, a zatim ukloni kontejner:

```bash
docker compose down
```
