# pobr-22L-RobertKaczmarski
Projekt rozpoznawanie logo sieci sklepów Lidl

## Pobranie i instalacja

```

# Klonowanie repozytorium

git clone https://github.com/hornisslaw/pobr-22L-RobertKaczmarski.git

cd pobr-22L-RobertKaczmarski

  

# Stworzenie i aktywacja wirtualnego środowiska

python -m venv venv

source ./venv/bin/activate

  

# instalacja potrzebnych bibliotek

pip install -r requirements.txt

```

  

## Opis parametrów i przykład wywołania programu

Parametry wywołania programu:

`-f` relatywna ścieżka do obrazu

`-r` wybór metody interpolacji do zmiany rozmiaru, możliwe metody to "bicubic", "nearest", bez podania tego parametru obraz będzie przetwarzany w oryginalnym rozmiarze

  

```
python main.py -f "images\\foto_1.jpg" -r "nearest"
```
